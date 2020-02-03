import re
from typing import Dict, List, Optional

import spacy
from spacy.tokens import Doc, Span
from spacy.util import filter_spans

from camphr.utils import SerializationMixin, get_doc_char_spans_list, merge_spans


@spacy.component(
    "multiple_regex_ruler", assigns=["doc.ents", "token.ent_type"], retokenizes=True
)
class MultipleRegexRuler(SerializationMixin):
    serialization_fields = ["patterns", "destructive", "merge"]
    _DEFAULT_LABEL = "matched"

    def __init__(
        self,
        patterns: Optional[Dict[str, str]] = None,
        destructive: bool = False,
        merge: bool = False,
        regex_flag: int = 0,
    ):
        self.patterns = patterns or {}
        self.destructive = destructive
        self.merge = merge
        self.regex_flag = regex_flag

    def require_model(self):
        assert self.patterns

    @property
    def labels(self) -> List[str]:
        self.require_model
        return list(self.patterns)  # type: ignore

    def __call__(self, doc: Doc) -> Doc:
        self.require_model()
        for label, pattern in self.patterns.items():  # type: ignore
            doc = self._proc(doc, pattern, label)
        return doc

    def _proc(self, doc: Doc, pattern: str, label: str) -> Doc:
        spans = self.get_spans(doc, pattern, label or self._DEFAULT_LABEL)
        doc.ents = filter_spans(tuple(spans) + doc.ents)  # type: ignore
        # TODO: https://stackoverflow.com/questions/59964767/mypy-incompatible-types-in-assignment-for-property-setter
        if self.merge:
            merge_spans(doc, spans)
        return doc

    def get_spans(self, doc: Doc, pattern: str, label: str) -> List[Span]:
        spans_ij = [m.span() for m in re.finditer(pattern, doc.text)]
        return get_doc_char_spans_list(
            doc, spans_ij, destructive=self.destructive, label=label
        )

    @classmethod
    def from_nlp(cls, *args, **kwargs) -> "MultipleRegexRuler":
        return cls()


@spacy.component(
    "regex_ruler", assigns=["doc.ents", "token.ent_type"], retokenizes=True
)
class RegexRuler(MultipleRegexRuler):
    def __init__(
        self,
        pattern: str = "",
        label: str = "",
        destructive: bool = False,
        merge: bool = False,
        name: str = "",
    ):
        self.patterns = {label: pattern}
        self.destructive = destructive
        self.merge = merge
        if name:
            self.name = name
        else:
            self.name = "regex_ruler_" + label
