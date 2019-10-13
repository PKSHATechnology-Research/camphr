from typing import Dict, Iterable, Optional, Tuple, Union

from spacy.tokens import Doc, Span
from typing_extensions import Protocol

from bedoner.utils import SerializationMixin

Match = Tuple[int, int, int]


class MatcherPipe(Protocol):
    def __call__(self, doc: Doc) -> Iterable[Match]:
        ...


class MatcherRuler(SerializationMixin):
    """Ruler with spacy.matcher.Matcher and PhraseMatcher."""

    def __init__(
        self,
        matcher: MatcherPipe,
        label: Optional[Union[int, str]] = None,
        matchid_to_label: Optional[Dict[int, Union[int, str]]] = None,
    ):
        self.matcher = matcher
        self.label = label
        self.matchid_to_label = matchid_to_label

    def __call__(self, doc: Doc) -> Doc:
        matches = self.matcher(doc)
        ents = []
        for match_id, start, end in matches:
            span = Span(doc, start, end, label=self.get_label(match_id))
            ents.append(span)
        doc.ents += tuple(ents)
        return doc

    def get_label(self, match_id: int) -> Union[int, str]:
        if self.label:
            return self.label
        if self.matchid_to_label:
            label = self.matchid_to_label.get(match_id)
            if label:
                return self.matchid_to_label[match_id]
        return match_id
