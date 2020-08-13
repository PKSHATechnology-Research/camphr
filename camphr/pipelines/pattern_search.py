"""Defines pattern search pipeline based on ahocorasik."""
from typing import (
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
    cast,
)

import ahocorasick
import spacy
import textspan
from spacy.tokens import Doc
from spacy.tokens.span import Span
from spacy.util import filter_spans
from typing_extensions import Literal

from camphr.utils import SerializationMixin, get_doc_char_span


@spacy.component("pattern_searcher")
class PatternSearcher(SerializationMixin):
    serialization_fields = [
        "model",
        "label_type",
        "custom_label",
        "custom_label_map",
        "destructive",
        "lemma",
        "lower",
        "cfg",
        "normalizer",
    ]

    def __init__(
        self,
        model: Optional[ahocorasick.Automaton] = None,
        label_type: Optional[
            Literal["custom_label_map", "custom_label", "value", "matched"]
        ] = None,
        custom_label: Optional[str] = None,
        custom_label_map: Optional[Dict[str, str]] = None,
        destructive: bool = False,
        lower: bool = False,
        normalizer: Optional[Callable[[str], str]] = None,
        lemma: bool = False,
        **cfg
    ):
        self.model = model
        self.label_type = label_type
        self.custom_label = custom_label
        self.custom_label_map = custom_label_map
        self._validate_label()
        self.destructive = destructive
        self.lower = lower
        self.cfg = cfg
        self.normalizer = normalizer
        self.lemma = lemma

        if custom_label:
            self.label_type = "custom_label"
            assert custom_label_map is None
            assert label_type == "custom_label" or label_type is None
        elif custom_label_map:
            self.label_type = "custom_label_map"
            assert custom_label is None
            assert label_type == "custom_label_map" or label_type is None
        if self.label_type is None:
            # default
            self.label_type = "matched"

    def _validate_label(self):
        if self.custom_label:
            self.label_type = "custom_label"
        elif self.custom_label_map:
            self.label_type = "custom_label_map"
        assert self.label_type != "custom_label_map" or self.custom_label_map
        assert self.label_type != "custom_label" or self.custom_label

    def get_label(self, item: str) -> str:
        if self.label_type == "matched":
            return "matched"
        if self.label_type == "value":
            return item
        if self.label_type == "custom_label":
            return cast(str, self.custom_label)
        if self.label_type == "custom_label_map":
            assert self.custom_label_map is not None
            return self.custom_label_map[item]

        raise ValueError("Internal Error")

    @classmethod
    def Model(cls, words: Iterable[str], **cfg) -> ahocorasick.Automaton:
        model = ahocorasick.Automaton()
        for word in words:
            model.add_word(word, word)
        model.make_automaton()
        return model

    @classmethod
    def from_words(cls, words: Iterable[str], **cfg) -> "PatternSearcher":
        model = cls.Model(words)
        return cls(model, **cfg)

    @classmethod
    def from_nlp(cls, *args, **kwargs) -> "PatternSearcher":
        return cls()

    def get_char_spans(self, text: str) -> Iterator[Tuple[int, int, str]]:
        self.require_model()
        for j, word in cast(ahocorasick.Automaton, self.model).iter(text):
            i = j - len(word) + 1
            yield i, j + 1, word

    def _to_text(self, doc: Union[Doc, Span]) -> str:
        if self.lemma:
            text = _to_lemma_text(doc)
        else:
            text = doc.text
        if self.lower:
            text = text.lower()
        if self.normalizer:
            text = self.normalizer(text)
        return text

    def __call__(self, doc: Doc) -> Doc:
        text = self._to_text(doc)
        _matches = list(self.get_char_spans(text))
        matches = _modify_spans(_matches, text, doc.text)
        spans = []
        for i, j, text in matches:
            if i is None or j is None:
                continue
            span = get_doc_char_span(
                doc,
                i,
                j,
                destructive=self.destructive,
                covering=not self.destructive,
                label=self.get_label(text),
            )
            if span and self._to_text(span) == text:
                spans.append(span)
        [s.text for s in spans]  # TODO: resolve the evaluation bug and remove this line
        ents = filter_spans(doc.ents + tuple(spans))
        doc.ents = tuple(ents)
        return doc


def _to_lemma_text(doc: Union[Doc, Span]) -> str:
    ret = ""
    for token in doc:
        ret += token.lemma_
        ret += token.whitespace_
    return ret


def _modify_spans(
    matches: List[Tuple[int, int, str]], text: str, original_text: str
) -> Iterator[Tuple[Optional[int], Optional[int], str]]:
    _spans = [(i, j) for i, j, _ in matches]
    spans = (
        (span[0][0], span[-1][1]) if len(span) > 0 else (None, None)
        for span in textspan.align_spans(_spans, text, original_text)
    )
    yield from ((*span, text[2]) for span, text in zip(spans, matches))
