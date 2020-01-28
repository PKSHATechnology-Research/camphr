from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple, cast

import numpy as np
import spacy
import torch
import torch.nn.functional as F
import transformers.modeling_bert
from spacy.gold import GoldParse
from spacy.language import Language
from spacy.pipeline import Pipe
from spacy.tokens import Doc
from transformers import BertConfig

from camphr.torch_utils import TorchPipe, add_loss_to_docs

from .model import TRANSFORMERS_MODEL
from .tokenizer import TRANSFORMERS_TOKENIZER, TrfTokenizer
from .utils import SerializationMixinForTrfTask, get_last_hidden_state_from_docs

MASKEDLM_PREDICTION = "maskedlm_prediction"
MASKEDLM_LABEL = "maskedlm_label"
BERT = "bert"


def get_maskedlm_labels(docs: List[Doc]) -> torch.Tensor:
    return docs[0].user_data[MASKEDLM_LABEL]


def set_maskedlm_labels(docs: List[Doc], labels: torch.Tensor):
    docs[0].user_data[MASKEDLM_LABEL] = labels


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

    @property  # type: ignore
    @lru_cache()
    def exclude_ids(self) -> np.ndarray:
        return np.array(self.model.all_special_ids)

    def __call__(self, doc: Doc):
        return doc

    def pipe(self, docs: Iterable[Doc], *args, **kwargs):
        return docs

    def update(self, docs: List[Doc], *args, **kwargs):
        self.require_model()
        inputs = TrfTokenizer.get_transformers_input(docs)
        assert inputs is not None
        input_ids = inputs.input_ids
        set_maskedlm_labels(docs, input_ids)

        masked_input_ids = input_ids.clone()
        target = torch.from_numpy(~np.isin(masked_input_ids, self.exclude_ids))
        mask_idx, replace_idx = self._choice_labels(masked_input_ids, target)

        masked_input_ids[mask_idx] = self.model.mask_token_id
        masked_input_ids[replace_idx] = torch.randint(
            low=max(self.model.all_special_ids),
            high=self.model.vocab_size,
            size=(cast(int, replace_idx.sum().cpu().item()),),
        )
        inputs.input_ids = masked_input_ids

    def _choice_labels(
        self, wordpieces: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        labels = torch.zeros_like(wordpieces, device=wordpieces.device)
        label = np.random.choice(3, target.sum().cpu().item(), p=self.p_dist)
        label = torch.from_numpy(label).to(device=wordpieces.device)
        labels[target] = label
        return (labels == 1, labels == 2)  # mask index, replace index

    def to_disk(self, path: Path, *args, **kwargs):
        # This component has nothing to be saved.
        # You should call `add_maskedlm` pipe after restoring.
        pass


@spacy.component(PIPES.bert_for_maskedlm)
class BertForMaskedLM(SerializationMixinForTrfTask, TorchPipe):
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
    wp = nlp.get_pipe(TRANSFORMERS_TOKENIZER)
    tokenizer = wp.model
    preprocessor = BertForMaskedLMPreprocessor(nlp.vocab, tokenizer)
    nlp.add_pipe(preprocessor, before=TRANSFORMERS_MODEL)
    bert = nlp.get_pipe(TRANSFORMERS_MODEL)
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
