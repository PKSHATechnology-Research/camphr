"""Defines transformers NER pipe"""
from typing import Dict, Iterable, Iterator, List, Sequence, Sized, cast

import spacy
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from spacy.gold import GoldParse, iob_to_biluo, spans_from_biluo_tags
from spacy.tokens import Doc

from camphr.pipelines.utils import (
    UNK,
    B,
    I,
    L,
    U,
    UserHooksMixin,
    beamsearch,
    construct_biluo_tag,
    correct_bio_tags,
    deconstruct_biluo_label,
)
from camphr_core.torch_utils import TorchPipe, add_loss_to_docs

from .auto import get_trf_config_cls
from .utils import (
    ATTRS,
    LABELS,
    EstimatorMixin,
    FromNLPMixinForTrfTask,
    LabelsMixin,
    SerializationMixinForTrfTask,
    TrfModelForTaskBase,
    get_dropout,
    get_last_hidden_state_from_docs,
)

NUM_LABELS = "num_labels"


class TrfTokenClassifier(TrfModelForTaskBase):
    """Head layer for classification task"""

    def __init__(self, config: transformers.PretrainedConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        dropout = get_dropout(config)
        hidden_size = config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, self.num_labels)
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(  # type:ignore
        self, x: torch.Tensor, mask: torch.Tensor = None, labels=None
    ) -> torch.Tensor:

        x = self.dropout(x)
        logits = self.classifier(x)
        return logits


class TrfForTokenClassificationBase(
    EstimatorMixin[torch.Tensor],
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

    def proc_model(self, docs: Iterable[Doc]) -> torch.Tensor:
        self.require_model()
        x = get_last_hidden_state_from_docs(docs)
        return self.model(x)

    @staticmethod
    def install_extensions():
        Doc.set_extension("tokens_logit", default=None, force=True)


TrfForTokenClassificationBase.install_extensions()
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
        if UNK in self.labels:
            return self.labels.index(UNK)
        return -1

    def compute_loss(
        self, docs: Sequence[Doc], golds: Sequence[GoldParse], outputs: torch.Tensor
    ) -> None:
        target = self._create_target_from_docs_golds(docs, golds, outputs)
        loss = F.cross_entropy(
            outputs.transpose(1, 2), target, ignore_index=self.ignore_label_index
        )
        add_loss_to_docs(docs, loss)

    def set_annotations(
        self, docs: Iterable[Doc], logits: torch.Tensor
    ) -> Iterable[Doc]:
        assert len(logits.shape) == 3  # (batch, length, nclass)
        id2label = self.labels

        for doc, logit in zip(docs, cast(Iterable, logits)):
            doc._.set("tokens_logit", logit)
            best_tags = get_best_tags(logit, id2label, self.k_beam)
            ents = [best_tags[a[0]] if len(a) else "O" for a in doc._.get(ATTRS.align)]
            biluo_ents = iob_to_biluo(ents)
            doc.ents = tuple(
                spacy.util.filter_spans(
                    doc.ents + tuple(spans_from_biluo_tags(doc, biluo_ents))
                )
            )
        return docs

    def _create_target_from_docs_golds(
        self, docs: Sequence[Doc], golds: Sequence[GoldParse], logits: torch.Tensor
    ) -> torch.Tensor:
        all_aligns = (doc._.get(ATTRS.align) for doc in docs)
        new_ners = (
            _convert_goldner(gold.ner or [], align)
            for gold, align in zip(golds, all_aligns)
        )
        return _create_target(new_ners, logits, self.ignore_label_index, self.label2id)

    def _get_nerlabel_from_gold(self, gold: GoldParse) -> Iterator[int]:
        if gold.ner is not None:
            for ner in gold.ner:
                yield self.label2id[self.convert_label(ner)]

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


def _convert_goldner(
    labels: Iterable[str], alignment: List[List[int]]
) -> Dict[int, str]:
    new_ner = {}
    prefixmap = {L: I, U: B}
    for label, align in zip(cast(Iterable[str], labels), alignment):
        prefix, type_ = deconstruct_biluo_label(label)
        new_prefix = prefixmap.get(prefix, prefix)
        for i in align:
            new_ner[i] = construct_biluo_tag(new_prefix, type_)
            if new_prefix == B:
                new_prefix = I
    return new_ner


def _create_target(
    new_ners: Iterable[Dict[int, str]],
    logits: torch.Tensor,
    ignore_index: int,
    label2id: Dict[str, int],
) -> torch.Tensor:
    batch_size, length, _ = logits.shape
    target = logits.new_full((batch_size, length), ignore_index, dtype=torch.long)
    for i, ners in enumerate(new_ners):
        idx_ners = [(i, label2id[ner]) for i, ner in ners.items()]
        if not idx_ners:
            continue
        idx, _ners = zip(*idx_ners)
        target[i, idx] = target.new_tensor(_ners)
    return target


def get_best_tags(logit: torch.Tensor, id2label: List[str], k_beam: int) -> List[str]:
    """Select best tags from logit based on beamsearch."""
    if k_beam > 1:
        logit = logit.softmax(-1)
    candidates = beamsearch(logit, k_beam)
    assert len(cast(Sized, candidates))
    best_tags: List[str] = []
    for cand in candidates:
        tags, is_correct = correct_bio_tags([id2label[j] for j in cast(Iterable, cand)])
        if is_correct:
            best_tags = tags
            break
        if not best_tags:
            best_tags = tags
    best_tags = [t if t != "-" else "O" for t in best_tags]
    return best_tags
