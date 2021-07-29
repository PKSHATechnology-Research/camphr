"""Named entity recognition pipe component for transformers"""
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Tuple
import textspan
from typing_extensions import TypedDict
from dataclasses import dataclass, asdict
from camphr.serde import SerDe, SerDeDataclassMixin
from pathlib import Path
from camphr.doc import Doc, DocProto, Ent, Token
import logging
from transformers import AutoTokenizer, AutoModelForTokenClassification
from camphr.nlp import Nlp


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
        output = list(map(int, self.model(**inputs).logits[0].argmax(1)))
        doc = _decode_bio(text, tokens, mask, output, self.model.config.id2label)
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


def _norm_tokens(tokens: List[str], mask: Sequence[int]) -> List[str]:
    """Replace special characters to _DUMMY to prevent incrrectly matching"""
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
    mask: Sequence[int],
    output: Sequence[int],
    id2label: Dict[int, str],
) -> Doc:
    assert len(output) == len(tokens)
    doc = Doc(text)
    labels = [id2label[i] for i in output]
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
