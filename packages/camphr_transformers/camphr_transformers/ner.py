"""Named entity recognition pipe component for transformers"""
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple

import textspan
import torch
from camphr.doc import Doc, Ent
from camphr.nlp import Nlp
from camphr.serde import SerDe, SerDeDataclassMixin
from transformers import AutoModelForTokenClassification, AutoTokenizer
from typing_extensions import TypedDict

logger = logging.getLogger(__name__)


class TrfEnt(TypedDict):
    entity_group: str
    score: float
    word: str
    start: int
    end: int


class Ner(Nlp, SerDe):
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        tokenizer_kwargs: Dict[str, Any] = {},
        model_kwargs: Dict[str, Any] = {},
    ):
        self.tokenizer, self.model = self._create_pipeline(
            pretrained_model_name_or_path, tokenizer_kwargs, model_kwargs
        )
        self.model.eval()
        self.tokenizer_kwargs = tokenizer_kwargs
        self.model_kwargs = model_kwargs

    @staticmethod
    def _create_pipeline(
        pretrained_model_name_or_path: str,
        tokenizer_kwargs: Dict[str, Any],
        model_kwargs: Dict[str, Any],
    ) -> Tuple[AutoTokenizer, AutoModelForTokenClassification]:
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, **tokenizer_kwargs
        )
        model = AutoModelForTokenClassification.from_pretrained(
            pretrained_model_name_or_path, **model_kwargs
        )
        return tokenizer, model

    def __call__(self, text: str) -> Doc:  # type: ignore
        inputs = self.tokenizer(text, return_tensors="pt")
        tokens: List[str] = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        mask: List[int] = list(
            map(
                int,
                self.tokenizer.get_special_tokens_mask(
                    inputs["input_ids"][0], already_has_special_tokens=True
                ),
            )
        )
        with torch.no_grad():
            output = self.model(**inputs).logits[0].argmax(1)
        labels: List[str] = [self.model.config.id2label[int(i)] for i in output]
        doc = _decode_bio(text, tokens, mask, labels)
        return doc

    # ser/de
    @dataclass
    class _SerdeMeta(SerDeDataclassMixin):
        tokenizer_kwargs: Dict[str, Any]
        model_kwargs: Dict[str, Any]
        FILENAME: ClassVar[str] = "camphr_meta.json"

    def to_disk(self, path: Path):
        meta = self._SerdeMeta(self.tokenizer_kwargs, self.model_kwargs)
        meta.to_disk(path)
        self.tokenizer.save_pretrained(str(path))
        self.model.save_pretrained(str(path))

    @classmethod
    def from_disk(cls, path: Path) -> "Ner":
        meta = cls._SerdeMeta.from_disk(path)
        return cls(str(path), **asdict(meta))


_DUMMY = "\u2581"


def _norm_tokens(tokens: List[str], mask: List[int]) -> List[str]:
    """Replace special characters to _DUMMY to prevent incrrectly matching"""
    assert len(tokens) == len(mask)
    ret: List[str] = []
    for token, mi in zip(tokens, mask):
        if int(mi):
            ret.append(_DUMMY)
        else:
            ret.append(token.replace(_DUMMY, ""))
    return ret


def _decode_bio(
    text: str,
    tokens: List[str],
    mask: List[int],
    labels: List[str],
) -> Doc:
    assert len(labels) == len(tokens)
    doc = Doc(text)

    # get Ent
    ents: List[Ent] = []
    cur_ent: Optional[Ent] = None
    tokens = _norm_tokens(tokens, mask)
    for span_lists, label in zip(textspan.get_original_spans(tokens, text), labels):
        if not span_lists:
            # special tokens should hit here
            continue
        l = span_lists[0][0]
        r = span_lists[-1][1]
        if label.startswith("I-") and cur_ent and cur_ent.label == label[2:]:
            # expand previous entity
            cur_ent.end_char = r
        elif label.startswith("I-") or label.startswith("B-"):
            # new entity
            if cur_ent:
                ents.append(cur_ent)
            cur_ent = Ent(l, r, doc, label=label[2:])
    if cur_ent:
        ents.append(cur_ent)
    doc.ents = ents
    return doc
