"""Defines tokenizer for transformers."""
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import spacy
import transformers
from camphr.pipelines.trf_utils import ATTRS
from camphr.utils import SerializationMixin
from spacy.pipeline import Pipe
from spacy.tokens import Doc


class PIPES:
    transformers_tokenizer = "transformers_tokenizer"


@spacy.component(PIPES.transformers_tokenizer)
class TransformersTokenizer(Pipe, SerializationMixin):
    _TRF_NAME = "trf_name"

    def __init__(self, model=True, **cfg):
        self.model = model
        self.trf_name = cfg.get("trf_name", "")

    @classmethod
    def from_nlp(cls, nlp, **cfg):
        return cls(**cfg)

    @classmethod
    def Model(
        cls, trf_name: str
    ) -> transformers.tokenization_utils.PreTrainedTokenizer:
        return transformers.AutoTokenizer.from_pretrained(trf_name)

    @classmethod
    def from_pretrained(cls, trf_name, **cfg):
        model = cls.Model(trf_name)
        return cls(model=model, trf_name=trf_name, **cfg)

    def to_disk(self, path: Path, *args, **kwargs):
        path.mkdir(exist_ok=True)
        model_save_path = path / self.trf_name
        model_save_path.mkdir(exist_ok=True)
        self.model.save_pretrained(str(model_save_path))
        with (path / "cfg.json").open("w") as f:
            cfg = {self._TRF_NAME: self.trf_name}
            json.dump(cfg, f)

    def from_disk(self, path: Path, *args, **kwargs):
        with (path / "cfg.json").open() as f:
            cfg = json.load(f)
            self.trf_name = cfg[self._TRF_NAME]
        self.model = self.Model(str(path / self.trf_name))

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
