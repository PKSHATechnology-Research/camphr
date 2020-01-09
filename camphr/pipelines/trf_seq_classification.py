import functools
from typing import Dict, Iterable, List, Optional, cast

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
    get_last_hidden_state_from_docs,
)
from camphr.pipelines.utils import UserHooksMixin
from camphr.torch_utils import add_loss_to_docs, goldcat_to_label
from overrides import overrides
from spacy.gold import GoldParse
from spacy.tokens import Doc
from transformers.modeling_utils import SequenceSummary

spacy.language.ENABLE_PIPELINE_ANALYSIS = True
NUM_SEQUENCE_LABELS = "num_sequence_labels"
LABELS = "labels"
TOP_LABEL = "top_label"
TOPK_LABELS = "topk_labels"


class TrfSequenceClassifier(TrfModelForTaskBase):
    """A thin layer for sequence classification task"""

    def __init__(self, config: TrfConfig):
        super().__init__(config)
        self.config = config
        assert hasattr(config, NUM_SEQUENCE_LABELS)
        self.num_labels = getattr(config, NUM_SEQUENCE_LABELS)
        self.sequence_summary = SequenceSummary(config)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.sequence_summary(x)
        logits = self.classifier(x)

        return logits


class TrfForSequenceClassificationBase(UserHooksMixin, TrfPipeForTaskBase):
    """Base class for sequence classification task (e.g. sentiment analysis).

    Requires `TrfModel` before this model in the pipeline.
    """

    model_cls = TrfSequenceClassifier

    @classmethod
    def Model(cls, **cfg) -> TrfSequenceClassifier:
        cfg.setdefault(TRF_CONFIG, {})
        assert cfg.get(LABELS), f"'{LABELS}' required"
        cfg[TRF_CONFIG][NUM_SEQUENCE_LABELS] = len(cfg.get(LABELS, []))
        model = super().Model(**cfg)
        return cast(TrfSequenceClassifier, model)

    @property
    def labels(self):
        return tuple(self.cfg.setdefault(LABELS, []))

    @property
    @functools.lru_cache()
    def label2id(self):
        return {v: i for i, v in enumerate(self.labels)}

    @property
    @functools.lru_cache()
    def label_weights(self) -> torch.Tensor:
        weights_map = self.cfg.get("label_weights")
        weights = torch.ones(len(self.label2id))
        if weights_map:
            assert len(weights_map) == len(self.label2id)
            for k, v in weights_map.items():
                weights[self.label2id[k]] = v
            return weights
        return weights

    @overrides
    def predict(self, docs: Iterable[Doc]) -> torch.Tensor:
        self.require_model()
        self.model.eval()
        with torch.no_grad():
            x = get_last_hidden_state_from_docs(docs)
            logits = self.model(x)
        assert len(logits.shape) == 2  # (len(docs), num_class)
        return logits

    @overrides
    def set_annotations(self, docs: Iterable[Doc], logits: torch.Tensor):
        probs = torch.softmax(logits, 1)
        for doc, prob in zip(docs, cast(Iterable, probs)):
            doc.cats = self.get_cats_from_prob(prob)

    def get_cats_from_prob(self, prob: torch.Tensor) -> Dict[str, float]:
        assert len(prob.shape) == 1
        return dict(zip(self.labels, prob.tolist()))

    def convert_label(self, label: str) -> str:
        fn = self.user_hooks.get(CONVERT_LABEL)
        if fn:
            return fn(label)
        return label

    def golds_to_tensor(self, golds: Iterable[GoldParse]) -> torch.Tensor:
        labels = (goldcat_to_label(gold.cats) for gold in golds)
        labels = (self.convert_label(label) for label in labels)
        targets = [self.label2id[label] for label in labels]
        return torch.tensor(targets, device=self.device)

    def update(self, docs: List[Doc], golds: Iterable[GoldParse]):
        assert isinstance(docs, list)
        self.require_model()
        self.model.train()

        x = get_last_hidden_state_from_docs(docs)
        logits = self.model(x)
        targets = self.golds_to_tensor(golds)
        weight = self.label_weights.to(device=self.device)

        loss = F.cross_entropy(logits, targets, weight=weight)
        add_loss_to_docs(docs, loss)


@spacy.component(
    "bert_seq_classifier",
    requires=[f"doc._.{ATTRS.last_hidden_state}"],
    assigns=["doc.cats"],
)
class BertForSequenceClassification(TrfForSequenceClassificationBase):
    trf_config_cls = trf.BertConfig


@spacy.component(
    "xlnet_seq_classifier",
    requires=[f"doc._.{ATTRS.last_hidden_state}"],
    assigns=["doc.cats"],
)
class XLNetForSequenceClassification(TrfForSequenceClassificationBase):
    trf_config_cls = trf.XLNetConfig


def _top_label(doc: Doc) -> Optional[str]:
    if not doc.cats:
        return None
    return max(doc.cats.items(), key=lambda x: x[1])[0]


def _topk_labels(doc: Doc, k: int) -> List[str]:
    if not doc.cats:
        return []
    return sorted(doc.cats.items(), key=lambda x: x[1], reverse=True)[:k]


Doc.set_extension(TOP_LABEL, getter=_top_label)
Doc.set_extension(TOPK_LABELS, method=_topk_labels)
