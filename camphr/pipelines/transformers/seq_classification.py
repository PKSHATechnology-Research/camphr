"""Defines transformers sequence classification pipe"""
import operator
from typing import Any, Dict, Iterable, List, Optional, Tuple, cast

import spacy
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from spacy.gold import GoldParse
from spacy.tokens import Doc
from transformers.modeling_utils import SequenceSummary

from camphr.pipelines.utils import UserHooksMixin
from camphr.torch_utils import TorchPipe, add_loss_to_docs, goldcat_to_label

from .auto import get_trf_config_cls
from .utils import (
    ATTRS,
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


TRANSFORMERS_SEQ_CLASSIFIER = "transformers_sequece_classifier"


@spacy.component(
    TRANSFORMERS_SEQ_CLASSIFIER,
    requires=[f"doc._.{ATTRS.last_hidden_state}"],
    assigns=["doc.cats"],
)
class TrfForSequenceClassification(
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

    def predict(self, docs: Iterable[Doc]) -> torch.Tensor:
        self.require_model()
        self.model.eval()
        with torch.no_grad():
            x = get_last_hidden_state_from_docs(docs)
            logits = self.model(x)
        assert len(logits.shape) == 2  # (len(docs), num_class)
        return logits

    def set_annotations(self, docs: Iterable[Doc], logits: torch.Tensor):
        probs = torch.softmax(logits, 1)
        for doc, prob in zip(docs, cast(Iterable, probs)):
            doc.cats = self.get_cats_from_prob(prob)

    def get_cats_from_prob(self, prob: torch.Tensor) -> Dict[str, float]:
        assert len(prob.shape) == 1
        return dict(zip(self.labels, prob.tolist()))

    def golds_to_tensor(self, golds: Iterable[GoldParse]) -> torch.Tensor:
        labels = (goldcat_to_label(gold.cats) for gold in golds)
        labels = (self.convert_label(label) for label in labels)
        targets = [self.label2id[label] for label in labels]
        return torch.tensor(targets, device=self.device)

    def update(  # type: ignore
        self, docs: List[Doc], golds: Iterable[GoldParse], **kwargs
    ):
        assert isinstance(docs, list)
        self.require_model()
        self.model.train()

        x = get_last_hidden_state_from_docs(docs)
        logits = self.model(x)
        targets = self.golds_to_tensor(golds)
        weight = self.label_weights.to(device=self.device)  # type: ignore

        loss = F.cross_entropy(logits, targets, weight=weight)
        add_loss_to_docs(docs, loss)


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
