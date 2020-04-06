"""Convert KNP dependency parsing result to spacy format."""
from typing import Iterable, Optional

import spacy
from spacy.tokens import Doc, Span

from camphr.pipelines.knp import KNP_USER_KEYS


@spacy.component("knp_dependency_parser", requires=("doc._.knp_tag_parent",))
def knp_dependency_parser(doc: Doc) -> Doc:
    tag_spans: Iterable[Span] = doc._.get(KNP_USER_KEYS.tag.spans)
    for tag in tag_spans:
        parent: Optional[Span] = tag._.get(KNP_USER_KEYS.tag.parent)
        if parent is not None:
            tag[0].head = parent[0]
        else:
            tag[0].head = tag[0]
        for p, c in zip(tag, tag[1:]):
            c.head = p
    doc.is_parsed = True
    return doc


def knp_dependency_parser_factory(*args, **kwargs):
    return knp_dependency_parser
