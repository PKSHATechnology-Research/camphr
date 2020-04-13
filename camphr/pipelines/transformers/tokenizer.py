"""Defines transformers tokenizer pipe."""
from typing import List, Optional, Sequence, Sized, cast

import spacy
import transformers
from spacy.pipeline import Pipe
from spacy.tokens import Doc
from spacy.vocab import Vocab

from .auto import get_trf_tokenizer_cls
from .utils import ATTRS, TransformersInput, TrfAutoMixin

TRANSFORMERS_TOKENIZER = "transformers_tokenizer"


@spacy.component(TRANSFORMERS_TOKENIZER)
class TrfTokenizer(TrfAutoMixin, Pipe):
    _TRF_NAME = "trf_name"
    _MODEL_CLS_GETTER = get_trf_tokenizer_cls

    def __init__(self, vocab: Vocab, model=True, **cfg):
        self.vocab = vocab
        self.model: transformers.PretrainedTokenizer = model
        self.cfg = cfg

    def predict(self, docs: Sequence[Doc]) -> TransformersInput:
        self.require_model()
        max_len = (
            None
            if self.model.max_len > max(len(doc.text) for doc in docs)
            else self.model.max_len
        )
        output = self.model.batch_encode_plus(
            [doc.text for doc in docs],
            add_special_tokens=True,
            return_tensors="pt",
            return_input_lengths=True,
            return_attention_masks=True,
            max_length=max_len,
            return_overflowing_tokens=True,
            return_special_tokens_mask=True,
            pad_to_max_length=True,
        )
        return TransformersInput(**output)

    def set_annotations(self, docs: Sequence[Doc], inputs: TransformersInput):
        self.set_transformers_input(docs, inputs)
        self._set_tokens(docs, inputs)

    def update(self, docs: Sequence[Doc], *args, **kwargs):  # type: ignore
        y = self.predict(docs)
        self.set_annotations(docs, y)

    @staticmethod
    def set_transformers_input(docs: Sequence[Doc], inputs: TransformersInput):
        if docs:
            docs[0]._.set(ATTRS.batch_inputs, inputs)

    @staticmethod
    def get_transformers_input(docs: Sequence[Doc]) -> Optional[TransformersInput]:
        if docs:
            output: TransformersInput = docs[0]._.get(ATTRS.batch_inputs)
            if len(docs) != len(cast(Sized, output.input_ids)):
                ValueError(
                    f"""Mismatching batch size during processing.
                `docs` has {len(docs)} length, but `batch_inputs` has {len(output.input_ids)}.
                Maybe each component is processing with a different batch size.
                """
                )
            return output
        return None

    def _set_tokens(self, docs: Sequence[Doc], inputs: TransformersInput) -> None:
        for doc, x in zip(docs, inputs):
            tokens = self.model.convert_ids_to_tokens(x.input_ids)
            doc._.set(ATTRS.tokens, tokens)
            doc._.set(ATTRS.cleaned_tokens, self._clean_special_tokens(tokens))

    def _clean_special_tokens(self, tokens: List[str]) -> List[str]:
        return [
            token if token not in self.model.all_special_tokens else ""
            for token in tokens
        ]
