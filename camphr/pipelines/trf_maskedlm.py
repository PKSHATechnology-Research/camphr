from functools import lru_cache
from typing import Any, Iterable, List, Optional

import numpy as np
import spacy
import torch
import torch.nn.functional as F
import transformers.modeling_bert
from spacy.gold import GoldParse
from spacy.language import Language
from spacy.pipeline import Pipe
from spacy.tokens import Doc
from spacy_transformers.util import ATTRS, get_tokenizer
from transformers import BertConfig

from camphr.pipelines.trf_utils import (
    TrfPipeForTaskBase,
    get_last_hidden_state_from_docs,
)
from camphr.pipelines.wordpiecer import PIPES as WP_PIPES
from camphr.torch_utils import add_loss_to_docs
from camphr.utils import zero_pad

MASKEDLM_PREDICTION = "maskedlm_prediction"
MASKEDLM_LABEL = "maskedlm_label"
BERT = "bert"


def get_maskedlm_labels(docs: List[Doc]) -> torch.tensor:
    return docs[0].user_data[MASKEDLM_LABEL]


def set_maskedlm_labels(docs: List[Doc], labels: np.ndarray):
    docs[0].user_data[MASKEDLM_LABEL] = torch.from_numpy(labels)


class BertOnlyMLMHead(transformers.modeling_bert.BertOnlyMLMHead):
    def __init__(self, config):
        super().__init__(config)
        self.config = config


class PIPES:
    bert_for_maskedlm_preprocessor = "bert_for_maskedlm_preprocessor"
    bert_for_maskedlm = "bert_for_maskedlm"


@spacy.component(PIPES.bert_for_maskedlm_preprocessor)
class BertForMaskedLMPreprocessor(Pipe):
    def __init__(self, vocab, model=True, **cfg):
        self.vocab = vocab
        self.cfg = cfg
        self.model = model
        self._exclude_ids: Optional[np.ndarray] = None

    @classmethod
    def Model(cls, **kwargs):
        return get_tokenizer(BERT).blank()

    @property
    def p_mask(self) -> float:
        return self.cfg.get(
            "p_mask", 0.15 * 0.8
        )  # ref: https://arxiv.org/pdf/1905.05583.pdf

    @property
    def p_replace(self) -> float:
        return self.cfg.get(
            "p_replace", 0.15 * 0.1
        )  # ref: https://arxiv.org/pdf/1905.05583.pdf

    @property
    def p_dist(self) -> List[float]:
        p_remain = 1 - self.p_mask - self.p_replace
        return [p_remain, self.p_mask, self.p_replace]

    @property
    @lru_cache()
    def exclude_ids(self) -> np.ndarray:
        return np.array(self.model.all_special_ids)

    def __call__(self, doc: Doc):
        return doc

    def pipe(self, docs: Iterable[Doc], *args, **kwargs):
        return docs

    def update(self, docs: List[Doc], *args, **kwargs):
        self.require_model()
        wordpieces = self._docs_to_wordpieces(docs)
        set_maskedlm_labels(docs, wordpieces)

        masked_wordpieces = wordpieces.copy()

        target = ~np.isin(masked_wordpieces, self.exclude_ids)
        mask_idx, replace_idx = self._choice_labels(masked_wordpieces, target)

        masked_wordpieces[mask_idx] = self.model.mask_token_id
        masked_wordpieces[replace_idx] = np.random.randint(
            max(self.model.all_special_ids),
            self.model.vocab_size,
            size=(replace_idx).sum(),
        )
        self._reset_wordpieces(docs, wordpieces)

    def _reset_wordpieces(self, docs: List[Doc], wordpieces: np.ndarray):
        for doc, wp in zip(docs, wordpieces):
            prev = doc._.get(ATTRS.word_pieces)
            doc._.set(ATTRS.word_pieces, wp[: len(prev)].tolist())

    def _docs_to_wordpieces(self, docs: List[Doc]) -> np.ndarray:
        wordpieces = [doc._.trf_word_pieces for doc in docs]
        return np.array(zero_pad(wordpieces, self.model.pad_token_id))

    def _choice_labels(self, wordpieces: np.ndarray, target: np.ndarray) -> np.ndarray:
        labels = np.zeros_like(wordpieces)
        label = np.random.choice(3, target.sum(), p=self.p_dist)
        labels[target] = label
        return labels == 1, labels == 2  # mask index, replace index


@spacy.component(PIPES.bert_for_maskedlm)
class BertForMaskedLM(TrfPipeForTaskBase):
    model_cls = BertOnlyMLMHead
    trf_config_cls = BertConfig

    def predict(self, docs: Iterable[Doc]):
        self.require_model()
        self.model.eval()
        x = get_last_hidden_state_from_docs(docs)
        with torch.no_grad():
            preds: torch.Tensor = self.model(x)
        assert len(preds.shape) == 3  # (b,l,h)
        return preds

    def set_annotations(self, docs: Iterable[Doc], preds: Any):
        for doc, pred in zip(docs, preds):
            doc.user_data[MASKEDLM_PREDICTION] = torch.max(pred)

    def update(self, docs: List[Doc], golds: Iterable[GoldParse]):
        self.require_model()
        self.model.train()
        x = get_last_hidden_state_from_docs(docs)
        preds: torch.Tensor = self.model(x)
        preds = preds.view(-1, self.model.config.vocab_size)
        targets = (
            get_maskedlm_labels(docs).view(-1)[: len(preds)].to(device=self.device)
        )
        loss = F.cross_entropy(preds, targets, ignore_index=-1)
        add_loss_to_docs(docs, loss)


def add_maskedlm_pipe(nlp: Language):
    """Add maskedlm pipe to nlp"""
    wp = nlp.get_pipe(WP_PIPES.transformers_wordpiecer)
    tokenizer = wp.model
    preprocessor = BertForMaskedLMPreprocessor(nlp.vocab, tokenizer)
    nlp.add_pipe(preprocessor, before=BERT)
    bert = nlp.get_pipe(BERT)
    config = bert.model.config
    model = BertOnlyMLMHead(config)
    pipe = BertForMaskedLM(nlp.vocab, model)
    nlp.add_pipe(pipe)


def remove_maskedlm_pipe(nlp: Language):
    """Remove maskedlm pipe from nlp"""
    targets = [PIPES.bert_for_maskedlm, PIPES.bert_for_maskedlm_preprocessor]
    for k in targets:
        if k in nlp.pipe_names:
            nlp.remove_pipe(k)
