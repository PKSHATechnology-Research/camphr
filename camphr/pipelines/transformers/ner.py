"""Module trf_ner defines pytorch transformers NER model

Models defined in this modules must be used with `camphr.pipelines.trf_model`'s model in `spacy.Language` pipeline
"""
from typing import Generator, Iterable, List, Sized, cast

import more_itertools
import spacy
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from overrides import overrides
from spacy.gold import GoldParse, spans_from_biluo_tags
from spacy.tokens import Doc

from camphr.pipelines.utils import UNK, UserHooksMixin, beamsearch, correct_biluo_tags
from camphr.torch_utils import TorchPipe, add_loss_to_docs

from .auto import get_trf_config_cls
from .utils import (
    ATTRS,
    LABELS,
    FromNLPMixinForTrfTask,
    LabelsMixin,
    SerializationMixinForTrfTask,
    TrfModelForTaskBase,
    get_dropout,
    get_last_hidden_state_from_docs,
)

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
    """

    model_cls = TrfTokenClassifier

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
    DEFAULT_BEAM_WIDTH = 5

    @property
    def ignore_label_index(self) -> int:
        if UNK.value in self.labels:
            return self.labels.index(UNK.value)
        return -1

    def update(self, docs: List[Doc], golds: List[GoldParse]):
        assert isinstance(docs, list)
        self.require_model()
        ignore_index = self.ignore_label_index
        x = get_last_hidden_state_from_docs(docs)
        logits = self.model(x)
        all_ners = (self._get_nerlabel_from_gold(gold) for gold in golds)
        all_aligns = (doc._.get(ATTRS.align) for doc in docs)
        target = _create_target(all_aligns, all_ners, logits, ignore_index)
        loss = F.cross_entropy(
            logits.transpose(1, 2), target, ignore_index=ignore_index
        )
        add_loss_to_docs(docs, loss)

    def _get_nerlabel_from_gold(self, gold: GoldParse) -> Generator[int, None, None]:
        if gold.ner is not None:
            for ner in gold.ner:
                yield self.label2id[self.convert_label(ner)]

    def set_annotations(
        self, docs: Iterable[Doc], logits: torch.Tensor
    ) -> Iterable[Doc]:
        """Modify a batch of documents, using pre-computed scores."""
        assert len(logits.shape) == 3  # (batch, length, nclass)
        id2label = self.labels

        for doc, logit in zip(docs, cast(Iterable, logits)):
            logit = self._extract_logit(logit, doc._.get(ATTRS.align))
            best_tags = get_best_tags(logit, id2label, self.k_beam)
            doc.ents = spacy.util.filter_spans(
                doc.ents + tuple(spans_from_biluo_tags(doc, best_tags))
            )
        return docs

    def _extract_logit(
        self, logit: torch.Tensor, alignment: List[List[int]]
    ) -> torch.Tensor:
        idx = [a[0] if len(a) > 0 else self.ignore_label_index for a in alignment]
        return logit[idx]

    @property
    def k_beam(self) -> int:
        return self.cfg.setdefault(self.K_BEAM, self.DEFAULT_BEAM_WIDTH)

    @k_beam.setter
    def k_beam(self, k: int):
        assert isinstance(k, int)
        self.cfg[self.K_BEAM] = k


def _create_target(
    all_aligns: Iterable[Iterable[Iterable[int]]],
    all_ners: Iterable[Iterable[int]],
    logits: torch.Tensor,
    ignore_index: int,
) -> torch.Tensor:
    # Why aren't this method accepting `docs` and `golds` directly? - testability.
    batch_size, length, _ = logits.shape
    target = logits.new_full((batch_size, length), ignore_index, dtype=torch.long)
    for i, (aligns, ners) in enumerate(zip(all_aligns, all_ners)):
        # use first wordpiece for each tokens
        idx = (more_itertools.first(a, None) for a in aligns)
        # drop elements where idx == None
        idx_ners = [(i, ner) for i, ner in zip(idx, ners) if i is not None]
        if not idx_ners:
            continue
        idx, ners = zip(*idx_ners)
        target[i, idx] = target.new_tensor(ners)
    return target


def get_best_tags(logit: torch.Tensor, id2label: List[str], k_beam: int) -> List[str]:
    """Select best tags from logit based on beamsearch."""
    candidates = beamsearch(logit.softmax(-1), k_beam)
    assert len(cast(Sized, candidates))
    best_tags = []
    for cand in candidates:
        tags, is_correct = correct_biluo_tags(
            [id2label[j] for j in cast(Iterable, cand)]
        )
        if is_correct:
            best_tags = tags
            break
        if not best_tags:
            best_tags = tags
    return best_tags
