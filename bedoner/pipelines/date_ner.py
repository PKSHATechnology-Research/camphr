"""Date extraction component for spacy pipeline. Partialy copied from https://github.com/PKSHATechnology/bedore-ner-module."""
from typing import Tuple

import regex as re
from spacy.language import Language
from spacy.tokens import Doc

import bedoner.ner_labels.labels_ontonotes as L
from bedoner.utils import SerializationMixin


class DateRuler(SerializationMixin):
    name = "bedoner_date_ruler"
    REGEXP_WAREKI_YMD = re.compile(
        r"(?:平成|昭和)(?:\d{1,2}|元)[/\\-年]\d{1,2}[/\\-月]\d{1,2}日?"
    )
    REGEXP_SEIREKI_YMD = re.compile(r"(\d{4})[/\\-年](\d{1,2})[/\\-月](\d{1,2})日?")
    LABEL = L.DATE

    serialization_fields = ["REGEXP_WAREKI_YMD", "REGEXP_SEIREKI_YMD", "LABEL"]

    @property
    def labels(self):
        return (self.LABEL,)

    def __call__(self, doc: Doc) -> Doc:
        """Extract date with regex"""
        for m in self.REGEXP_WAREKI_YMD.finditer(doc.text):
            span = doc.char_span(*m.span(), label=self.LABEL)
            doc.ents = doc.ents + (span,)

        for m in self.REGEXP_SEIREKI_YMD.finditer(doc.text):
            if self._is_valid_seireki(m.groups()):
                span = doc.char_span(*m.span(), label=self.LABEL)
                doc.ents = doc.ents + (span,)
        return doc

    @staticmethod
    def _is_valid_seireki(found_expressions: Tuple[str, str, str, str]) -> bool:
        year = int(found_expressions[0])
        month = int(found_expressions[1])
        day = int(found_expressions[2])
        return (1500 <= year <= 2100) and (1 <= month <= 12) and (1 <= day <= 31)


Language.factories[DateRuler.name] = DateRuler
