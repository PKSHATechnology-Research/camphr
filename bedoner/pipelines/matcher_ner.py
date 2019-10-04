from typing_extensions import Protocol
from typing import Dict, Iterable, Optional, Tuple, Union
from spacy.tokens import Doc, Span
from bedoner.utils import SerializationMixin


Match = Tuple[int, int, int]


class MatcherPipe(Protocol):
    def __call__(self, doc: Doc) -> Iterable[Match]:
        ...


class MatcherNer(SerializationMixin):
    """Ner with spacy.matcher.Matcher and PhraseMatcher."""

    def __init__(
        self,
        matcher: MatcherPipe,
        label: Optional[Union[int, str]] = None,
        matchid_to_label: Optional[Dict[int, Union[int, str]]] = None,
    ):
        assert label or matchid_to_label
        self.matcher = matcher
        self.label = label
        self.matchid_to_label = matchid_to_label

    def __call__(self, doc: Doc) -> Doc:
        matches: Iterable[Match] = self.matcher(doc)
        ents = []
        for match_id, start, end in matches:
            span = Span(doc, start, end, label=self.get_label(match_id))
            ents.append(span)
        doc.ents += tuple(ents)
        return doc

    def get_label(self, match_id: int) -> Union[int, str]:
        if self.label:
            return self.label
        return self.matchid_to_label[match_id]
