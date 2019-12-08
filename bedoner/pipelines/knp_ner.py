from functools import reduce
from typing import Any, List

import spacy
from bedoner.consts import KEY_KNP_ENT, KEY_KNP_ENT_IOB
from spacy.tokens import Doc, Span, Token


@spacy.component(
    "knp_entity_extractor",
    assigns=["doc.ents"],
    requires=[f"token._.{KEY_KNP_ENT}", f"token._.{KEY_KNP_ENT_IOB}"],
)
class KnpEntityExtractor:
    """A spacy pipline component to extract named entity from knp output.

    This component must be used with `bedoner.lang.knp.Japanse`
    """

    name = "knp_entity_extractor"

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, doc: Doc) -> Doc:
        ents = []
        for text, start, end in reduce(self._extract_ents, doc, []):
            ents.append(Span(doc, start, end, label=text))
        doc.ents += tuple(ents)
        return doc

    @staticmethod
    def _extract_ents(ents: List[List[Any]], token: Token) -> List[List[Any]]:
        if token._.get(KEY_KNP_ENT_IOB) == "B":
            ents.append([token._.get(KEY_KNP_ENT), token.i, token.i + 1])
        elif token._.get(KEY_KNP_ENT_IOB) == "I":
            ents[-1][2] = token.i + 1

        return ents
