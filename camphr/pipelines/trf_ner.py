"""Module trf_ner defines pytorch transformers NER model

Models defined in this modules must be used with `camphr.pipelines.trf_model`'s model in `spacy.Language` pipeline
"""
import functools
from typing import Iterable, List, Sized, cast

import spacy
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers as trf
from camphr.pipelines.trf_utils import (
    ATTRS,
    CONVERT_LABEL,
    TRF_CONFIG,
    TrfConfig,
    TrfModelForTaskBase,
    TrfPipeForTaskBase,
    get_dropout,
    get_last_hidden_state_from_docs,
)
from camphr.pipelines.utils import (
    UNK,
    UserHooksMixin,
    beamsearch,
    correct_biluo_tags,
    merge_entities,
)
from camphr.torch_utils import add_loss_to_docs
from overrides import overrides
from spacy.gold import GoldParse, spans_from_biluo_tags
from spacy.tokens import Doc, Token

CLS_LOGIT = "cls_logit"
LABELS = "labels"
NUM_LABELS = "num_labels"


class TrfTokenClassifier(TrfModelForTaskBase):
    """A thin layer for classification task"""

    def __init__(self, config: TrfConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        dropout = get_dropout(config)
        hidden_size = config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, self.num_labels)
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor = None, labels=None
    ) -> torch.Tensor:

        x = self.dropout(x)
        logits = self.classifier(x)

        return logits


class TrfForTokenClassificationBase(UserHooksMixin, TrfPipeForTaskBase):
    """Base class for token classification task (e.g. Named entity recognition).

    Requires `TrfModel` before this model in the pipeline to use this model.

    Notes:
        `Token._.cls_logit` is set and stored the output of this model into. This is usefule to calculate the probability of the classification.
        `Doc._.cls_logit` is set and stored the output of this model into.
    """

    model_cls = TrfTokenClassifier

    @staticmethod
    def install_extensions():
        token_exts = [CLS_LOGIT]
        doc_exts = [CLS_LOGIT]
        for ext in token_exts:
            if Token.get_extension(ext) is None:
                Token.set_extension(ext, default=None)
        for ext in doc_exts:
            if Doc.get_extension(ext) is None:
                Doc.set_extension(ext, default=None)

    @classmethod
    def Model(cls, **cfg) -> TrfTokenClassifier:
        assert cfg.get(LABELS)
        cfg.setdefault(TRF_CONFIG, {})
        cfg[TRF_CONFIG][NUM_LABELS] = len(cfg.get(LABELS, []))
        model = super().Model(**cfg)
        assert model.config.num_labels == len(cfg[LABELS])
        return cast(TrfTokenClassifier, model)

    @property
    def labels(self):
        return tuple(self.cfg.setdefault(LABELS, []))

    @property
    @functools.lru_cache()
    def label2id(self):
        return {v: i for i, v in enumerate(self.labels)}

    @overrides
    def predict(self, docs: Iterable[Doc]) -> torch.Tensor:
        self.require_model()
        self.model.eval()
        with torch.no_grad():
            x = get_last_hidden_state_from_docs(docs)
            logits = self.model(x)
        return logits


class TrfForNamedEntityRecognitionBase(TrfForTokenClassificationBase):
    """Named entity recognition component with pytorch-transformers."""

    K_BEAM = "k_beam"

    @property
    def ignore_label_index(self) -> int:
        if UNK in self.labels:
            return self.labels.index(UNK)
        return -1

    def update(self, docs: List[Doc], golds: Iterable[GoldParse]):
        assert isinstance(docs, list)
        self.require_model()
        label2id = self.label2id
        ignore_index = self.ignore_label_index
        x = get_last_hidden_state_from_docs(docs)
        logits = self.model(x)
        length = logits.shape[1]
        loss = x.new_tensor(0.0)
        for doc, gold, logit in zip(docs, golds, logits):
            # use first wordpiece for each tokens
            idx = []
            convert_hook = self.user_hooks.get(CONVERT_LABEL, None)
            if convert_hook:
                ners = [convert_hook(ner) for ner in gold.ner]
            else:
                ners = list(gold.ner)
            for i, align in enumerate(doc._.transformers_align):
                if len(align):
                    a = align[0]
                    if a >= length:
                        # This is not a bug. `length` can be shorter than the pieces because of `max_length` of trf model.
                        break
                    idx.append(a)
                else:
                    ners[i] = UNK.value  # avoid calculate loss
                    idx.append(-1)
            pred = logit[idx]
            target = torch.tensor(
                [label2id[ner] for ner in ners[: len(pred)]], device=self.device
            )
            loss += F.cross_entropy(pred, target, ignore_index=ignore_index)
            doc._.cls_logit = logit
        add_loss_to_docs(docs, loss)

    def set_annotations(
        self, docs: Iterable[Doc], logits: torch.Tensor
    ) -> Iterable[Doc]:
        """Modify a batch of documents, using pre-computed scores."""
        assert len(logits.shape) == 3  # (batch, length, nclass)
        id2label = self.labels

        for doc, logit in zip(docs, cast(Iterable, logits)):
            logit = self._extract_logit(logit, doc._.get(ATTRS.align))
            candidates = beamsearch(logit, self.k_beam)
            assert len(cast(Sized, candidates))
            best_tags = None
            for cand in candidates:
                tags, is_correct = correct_biluo_tags(
                    [id2label[j] for j in cast(Iterable, cand)]
                )
                if is_correct:
                    best_tags = tags
                    break
                if best_tags is None:
                    best_tags = tags
            doc.ents = merge_entities(doc.ents, spans_from_biluo_tags(doc, best_tags))
        return docs

    def _extract_logit(
        self, logit: torch.Tensor, alignment: List[List[int]]
    ) -> torch.Tensor:
        idx = [a[0] for a in alignment if len(a) > 0]
        return logit[idx]

    @property
    def k_beam(self) -> int:
        return self.cfg.setdefault(self.K_BEAM, 10)

    @k_beam.setter
    def k_beam(self, k: int):
        assert isinstance(k, int)
        self.cfg[self.K_BEAM] = k


@spacy.component(
    "bert_ner", requires=[f"doc._.{ATTRS.last_hidden_state}"], assigns=["doc.ents"]
)
class BertForNamedEntityRecognition(TrfForNamedEntityRecognitionBase):
    trf_config_cls = trf.BertConfig


@spacy.component(
    "xlnet_ner", requires=[f"doc._.{ATTRS.last_hidden_state}"], assigns=["doc.ents"]
)
class XLNetForNamedEntityRecognition(TrfForNamedEntityRecognitionBase):
    trf_config_cls = trf.XLNetConfig


TrfForTokenClassificationBase.install_extensions()
