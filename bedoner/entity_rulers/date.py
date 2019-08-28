from spacy.tokens.doc import Doc
import regex as re
from typing import List, Tuple, Union  # noqa # pylint: disable=unused-import
from .labels import L

REGEXP_WAREKI_YMD = re.compile(r"(?:平成|昭和)(?:\d{1,2}|元)[/\\-年]\d{1,2}[/\\-月]\d{1,2}日?")
REGEXP_SEIREKI_YMD = re.compile(r"(\d{4})[/\\-年](\d{1,2})[/\\-月](\d{1,2})日?")


def _is_valid_seireki(found_expressions: Tuple[str, str, str, str]) -> bool:
    year = int(found_expressions[0])
    month = int(found_expressions[1])
    day = int(found_expressions[2])
    return (1500 <= year <= 2100) and (1 <= month <= 12) and (1 <= day <= 31)


def date_ruler(doc: Doc) -> Doc:
    for m in REGEXP_WAREKI_YMD.finditer(doc.text):
        span = doc.char_span(*m.span(), label=L.DATE)
        doc.ents = doc.ents + (span,)

    for m in REGEXP_SEIREKI_YMD.finditer(doc.text):
        if _is_valid_seireki(m.groups()):
            span = doc.char_span(*m.span(), label=L.DATE)
            doc.ents = doc.ents + (span,)
    return doc
