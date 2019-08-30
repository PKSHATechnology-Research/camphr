from typing import List, Any
from functools import reduce
from spacy.tokens import Doc, Span, Token
from bedoner.consts import KEY_KNP_ENT, KEY_KNP_ENT_IOB


def _extract_ents(ents: List[List[Any]], token: Token) -> List[List[Any]]:
    if token._.get(KEY_KNP_ENT_IOB) == "B":
        ents.append([token._.get(KEY_KNP_ENT), token.i, token.i + 1])
        return ents
    if token._.get(KEY_KNP_ENT_IOB) == "I":
        ents[-1][2] = token.i + 1
        return ents
    return ents


def knp_entity_extractor(doc: Doc) -> Doc:
    """A spacy pipline component to extract named entity"""
    ents = []
    for text, start, end in reduce(_extract_ents, doc, []):
        ents.append(Span(doc, start, end, label=text))
    doc.ents += tuple(ents)
    return doc
