"""Defines tokenizer for transformers."""
from typing import Dict, Iterable, List, Sequence

import spacy
from camphr.pipelines.trf_auto import get_trf_tokenizer_cls
from camphr.pipelines.trf_utils import ATTRS, TrfAutoMixin
from spacy.pipeline import Pipe
from spacy.tokens import Doc
from spacy.vocab import Vocab


class PIPES:
    transformers_tokenizer = "transformers_tokenizer"


@spacy.component(PIPES.transformers_tokenizer)
class TransformersTokenizer(TrfAutoMixin, Pipe):
    _TRF_NAME = "trf_name"
    _MODEL_CLS_GETTER = get_trf_tokenizer_cls

    def __init__(self, vocab: Vocab, model=True, **cfg):
        self.vocab = vocab
        self.model = model
        self.cfg = cfg

    def update(self, docs: Sequence[Doc], *args, **kwargs) -> List[Doc]:
        return [self(doc) for doc in docs]

    def predict(self, docs: Iterable[Doc]) -> List[Dict[str, List[int]]]:
        return [
            self.model.encode_plus(doc.text, max_length=self.model.max_len)
            for doc in docs
        ]

    def set_annotations(
        self, docs: Sequence[Doc], inputs_list: List[Dict[str, List[int]]]
    ):
        for doc, inputs in zip(docs, inputs_list):
            doc._.set(ATTRS.token_ids, inputs["input_ids"])
            doc._.set(ATTRS.token_type_ids, inputs["token_type_ids"])
            doc._.set(ATTRS.attention_mask, inputs["attention_mask"])

            tokens = self.model.convert_ids_to_tokens(inputs["input_ids"])
            doc._.set(ATTRS.tokens, tokens)
            doc._.set(ATTRS.cleaned_tokens, self._clean_special_tokens(tokens))

    def _clean_special_tokens(self, tokens: List[str]) -> List[str]:
        return [
            token if token not in self.model.all_special_tokens else ""
            for token in tokens
        ]
