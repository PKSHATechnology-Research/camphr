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
from transformers.pipelines.token_classification import TokenClassificationPipeline
from transformers.pipelines import pipeline as trf_pipeline
from camphr.nlp import Nlp

_SPECIAL_TOKENS_MASK = "special_tokens_mask"
_DUMMY = "\u2581"

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

    def __call__(self, text: str) -> DocProto[Token, Ent]:
        inputs = self.tokenizer(
            text, return_tensors="pt", return_special_tokens_mask=True
        )
        tokens: List[str] = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        mask: Sequence[int] = inputs[_SPECIAL_TOKENS_MASK][0]
        del inputs[_SPECIAL_TOKENS_MASK]
        output = self.model(**inputs).logits[0].argmax(1)
        assert len(output.shape) == 1, output.shape
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

    @classmethod
    def from_disk(cls, path: Path) -> "Ner":
        meta = cls._SerdeMeta.from_disk(path)
        return cls(str(path), **asdict(meta))


def _decode_label(
    mask: Sequence[int], output: Sequence[int], id2label: Dict[int, str]
) -> List[str]:
    ret: List[str] = []
    assert len(mask) == len(output)
    for mi, oi in zip(mask, output):
        if int(mi):
            continue
        ret.append(id2label[int(oi)])
    return ret


def _decode_bio(
    text: str,
    tokens: List[str],
    mask: Sequence[int],
    output: Sequence[int],
    id2label: Dict[int, str],
) -> Doc:
    doc = Doc(text)
    labels = _decode_label(mask, output, id2label)
    assert len(labels) == len(tokens), (labels, tokens)
    new_tokens: List[Token] = []
    ents: List[Ent] = []
    cur_ent: Optional[Ent] = None
    for span_lists, token, label in zip(
        textspan.get_original_spans(tokens, text), tokens, labels
    ):
        if not span_lists:
            logger.warn(f"Ignore token: {token}")
            continue
        l = span_lists[0][0]
        r = span_lists[-1][1]
        new_tokens.append(Token(start_char=l, end_char=r, doc=doc))
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
    doc.tokens = new_tokens
    return doc
