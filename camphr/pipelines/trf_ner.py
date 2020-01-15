"""Module trf_ner defines pytorch transformers NER model

Models defined in this modules must be used with `camphr.pipelines.trf_model`'s model in `spacy.Language` pipeline
"""
from typing import Iterable, List, Sized, cast

import more_itertools
import spacy
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from camphr.pipelines.trf_utils import (
    ATTRS,
    LABELS,
    FromNLPMixinForTrfTask,
    LabelsMixin,
    SerializationMixinForTrfTask,
    TrfModelForTaskBase,
    get_dropout,
    get_last_hidden_state_from_docs,
)
from camphr.pipelines.utils import UNK, UserHooksMixin, beamsearch, correct_biluo_tags
from camphr.torch_utils import TorchPipe, add_loss_to_docs
from overrides import overrides
from spacy.gold import GoldParse, spans_from_biluo_tags
from spacy.tokens import Doc, Token

from .trf_auto import get_trf_config_cls

CLS_LOGIT = "cls_logit"
NUM_LABELS = "num_labels"


class TrfTokenClassifier(TrfModelForTaskBase):
    """A thin layer for classification task"""

    def __init__(self, config: transformers.PretrainedConfig):
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


class TrfForTokenClassificationBase(
    LabelsMixin,
    FromNLPMixinForTrfTask,
    UserHooksMixin,
    SerializationMixinForTrfTask,
    TorchPipe,
):
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
    def Model(cls, trf_name_or_path: str, **cfg) -> TrfTokenClassifier:
        config = get_trf_config_cls(trf_name_or_path).from_pretrained(trf_name_or_path)
        setattr(config, NUM_LABELS, len(cfg[LABELS]))
        return TrfTokenClassifier(config)

    @overrides
    def predict(self, docs: Iterable[Doc]) -> torch.Tensor:
        self.require_model()
        self.model.eval()
        with torch.no_grad():
            x = get_last_hidden_state_from_docs(docs)
            logits = self.model(x)
        return logits


TRANSFORMERS_NER = "transformers_ner"


@spacy.component(
    TRANSFORMERS_NER,
    requires=[f"doc._.{ATTRS.last_hidden_state}"],
    assigns=["doc.ents"],
)
class TrfForNamedEntityRecognition(TrfForTokenClassificationBase):
    """Named entity recognition component with pytorch-transformers."""

    K_BEAM = "k_beam"

    @property
    def ignore_label_index(self) -> int:
        if UNK.value in self.labels:
            return self.labels.index(UNK.value)
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
            ners = self._get_nerlabel_from_gold(gold)
            # use first wordpiece for each tokens
            idx = (more_itertools.first(a, -1) for a in doc._.get(ATTRS.align))
            idx = [a for a in idx if a < length]
            ners = (label2id[ner] for ner in ners)
            ners = [ner if a != -1 else ignore_index for ner, a in zip(ners, idx)]
            pred = logit[idx]
            target = torch.tensor(ners, device=self.device)
            loss += F.cross_entropy(pred, target, ignore_index=ignore_index)
            doc._.cls_logit = logit
        add_loss_to_docs(docs, loss)

    def _get_nerlabel_from_gold(self, gold: GoldParse) -> List[str]:
        ners = [self.convert_label(ner) for ner in gold.ner]
        return ners

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
            doc.ents = spacy.util.filter_spans(
                doc.ents + tuple(spans_from_biluo_tags(doc, best_tags))
            )
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


TrfForTokenClassificationBase.install_extensions()
