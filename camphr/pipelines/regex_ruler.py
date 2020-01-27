import re
from typing import AnyStr, Dict, List, Pattern

import spacy
from spacy.tokens import Doc, Span
from spacy.util import filter_spans

from camphr.utils import SerializationMixin, get_doc_char_spans_list, merge_spans


@spacy.component(
    "multiple_regex_ruler", assigns=["doc.ents", "token.ent_type"], retokenizes=True
)
class MultipleRegexRuler(SerializationMixin):
    serialization_fields = ["patterns", "destructive", "merge"]

    def __init__(
        self,
        patterns: Dict[str, AnyStr],
        destructive: bool = False,
        merge: bool = False,
        regex_flag: int = 0,
    ):
        self.patterns = self.compile(patterns, regex_flag)
        self.destructive = destructive
        self.merge = merge
        self.regex_flag = regex_flag

    def compile(self, patterns: Dict[str, AnyStr], flag: int) -> Dict[str, Pattern]:
        return {k: re.compile(v, flag) for k, v in patterns.items()}

    @property
    def labels(self) -> List[str]:
        return list(self.patterns)

    def __call__(self, doc: Doc) -> Doc:
        for label, pattern in self.patterns.items():
            doc = self._proc(doc, pattern, label)
        return doc

    def _proc(self, doc: Doc, pattern: Pattern, label: str) -> Doc:
        spans = self.get_spans(doc, pattern, label)
        doc.ents = filter_spans(doc.ents + tuple(spans))
        if self.merge:
            merge_spans(doc, spans)
        return doc

    def get_spans(self, doc: Doc, pattern: Pattern, label: str) -> List[Span]:
        spans_ij = [m.span() for m in pattern.finditer(doc.text)]
        return get_doc_char_spans_list(
            doc, spans_ij, destructive=self.destructive, label=label
        )


@spacy.component(
    "regex_ruler", assigns=["doc.ents", "token.ent_type"], retokenizes=True
)
class RegexRuler(MultipleRegexRuler):
    def __init__(
        self,
        pattern,
        label: str,
        destructive: bool = False,
        merge: bool = False,
        name: str = "",
    ):
        self.patterns = {label: re.compile(pattern)}
        self.destructive = destructive
        self.merge = merge
        if name:
            self.name = name
        else:
            self.name = "regex_ruler_" + label
