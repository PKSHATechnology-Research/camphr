"""Defines transformers sequence classification pipe"""
import operator
from typing import Any, Iterable, List, Optional, Sequence, Tuple, cast

import spacy
from spacy.gold import GoldParse
import spacy.language
from spacy.tokens import Doc
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers.modeling_utils import SequenceSummary

from camphr.pipelines.utils import UserHooksMixin
from camphr_core.torch_utils import TorchPipe, add_loss_to_docs, goldcat_to_label

from .auto import get_trf_config_cls
from .utils import (
    ATTRS,
    EstimatorMixin,
    FromNLPMixinForTrfTask,
    LabelsMixin,
    SerializationMixinForTrfTask,
    TrfModelForTaskBase,
    get_last_hidden_state_from_docs,
)

spacy.language.ENABLE_PIPELINE_ANALYSIS = True
NUM_SEQUENCE_LABELS = "num_sequence_labels"
LABELS = "labels"
TOP_LABEL = "top_label"
TOPK_LABELS = "topk_labels"


class TrfSequenceClassifier(TrfModelForTaskBase):
    """Head layer for sequence classification task"""

    def __init__(self, config: transformers.PretrainedConfig):
        super().__init__(config)
        self.config = config
        assert hasattr(config, NUM_SEQUENCE_LABELS)
        self.num_labels = getattr(config, NUM_SEQUENCE_LABELS)
        self.sequence_summary = SequenceSummary(config)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        x = self.sequence_summary(x)
        logits = self.classifier(x)

        return logits


class TrfForSequenceClassificationBase(
    EstimatorMixin[torch.Tensor],
    LabelsMixin,
    FromNLPMixinForTrfTask,
    UserHooksMixin,
    SerializationMixinForTrfTask,
    TorchPipe,
):
    """Base class for sequence classification task (e.g. sentiment analysis).

    Requires `TrfModel` before this model in the pipeline.
    """

    model_cls = TrfSequenceClassifier

    @classmethod
    def Model(cls, trf_name_or_path: str, **cfg) -> TrfSequenceClassifier:
        config = get_trf_config_cls(trf_name_or_path).from_pretrained(trf_name_or_path)
        if LABELS in cfg:
            setattr(config, NUM_SEQUENCE_LABELS, len(cfg[LABELS]))
        return TrfSequenceClassifier(config)

    def proc_model(self, docs: Iterable[Doc]) -> torch.Tensor:
        self.require_model()
        x = get_last_hidden_state_from_docs(docs)
        logits = self.model(x)
        assert len(logits.shape) == 2  # (len(docs), num_class)
        return logits

    def set_cats_to_docs(self, docs: Iterable[Doc], probs: torch.Tensor):
        for doc, prob in zip(docs, cast(Iterable, probs)):
            doc.cats = dict(zip(self.labels, prob.tolist()))


TRANSFORMERS_SEQ_CLASSIFIER = "transformers_sequence_classifier"


@spacy.component(
    TRANSFORMERS_SEQ_CLASSIFIER,
    requires=[f"doc._.{ATTRS.last_hidden_state}"],
    assigns=["doc.cats"],
)
class TrfForSequenceClassification(TrfForSequenceClassificationBase):
    """Sequence classification task (e.g. sentiment analysis)."""

    def golds_to_tensor(self, golds: Iterable[GoldParse]) -> torch.Tensor:
        labels = (goldcat_to_label(gold.cats) for gold in golds)
        labels = (self.convert_label(label) for label in labels)
        targets = [self.label2id[label] for label in labels]
        return torch.tensor(targets, device=self.device)

    def compute_loss(
        self, docs: Sequence[Doc], golds: Sequence[GoldParse], outputs: torch.Tensor
    ) -> None:
        self.require_model()
        targets = self.golds_to_tensor(golds)
        weight = self.label_weights.to(device=self.device)  # type: ignore

        loss = F.cross_entropy(outputs, targets, weight=weight)
        add_loss_to_docs(docs, loss)

    def set_annotations(self, docs: Iterable[Doc], logits: torch.Tensor):
        probs = torch.softmax(logits, 1)
        self.set_cats_to_docs(docs, probs)


TRANSFORMERS_MULTILABEL_SEQ_CLASSIFIER = "transformers_multilabel_sequence_classifier"


@spacy.component(
    TRANSFORMERS_MULTILABEL_SEQ_CLASSIFIER,
    requires=[f"doc._.{ATTRS.last_hidden_state}"],
    assigns=["doc.cats"],
)
class TrfForMultiLabelSequenceClassification(TrfForSequenceClassificationBase):
    """Multi labels sequence classification task (e.g. sentiment analysis)."""

    def compute_loss(
        self, docs: Sequence[Doc], golds: Sequence[GoldParse], outputs: torch.Tensor
    ) -> None:
        self.require_model()
        targets = outputs.new_tensor(
            [[gold.cats[label] for label in self.labels] for gold in golds]
        )
        loss = F.binary_cross_entropy_with_logits(outputs, targets)
        add_loss_to_docs(docs, loss)

    def set_annotations(self, docs: Iterable[Doc], logits: torch.Tensor):
        probs = torch.sigmoid(logits)
        self.set_cats_to_docs(docs, probs)


def _top_label(doc: Doc) -> Optional[str]:
    if not doc.cats:
        return None
    return max(doc.cats.items(), key=operator.itemgetter(1))[0]


def _topk_labels(doc: Doc, k: int) -> List[Tuple[str, Any]]:
    if not doc.cats:
        return []
    return sorted(doc.cats.items(), key=operator.itemgetter(1), reverse=True)[:k]


Doc.set_extension(TOP_LABEL, getter=_top_label, force=True)
Doc.set_extension(TOPK_LABELS, method=_topk_labels, force=True)
