from typing import Tuple

import regex as re
from spacy.language import Language
from spacy.tokens import Doc

import bedoner.ner_labels.labels_ontonotes as L
from bedoner.pipelines.utils import merge_entities
from bedoner.utils import SerializationMixin, destruct_token


class RegexRuler(SerializationMixin):
    def __init__(
        self,
        pattern,
        label: str,
        destructive: bool = False,
        merge: bool = False,
        name: str = "",
    ):
        self.pattern = re.compile(pattern)
        self.destructive = destructive
        self.label = label
        self.labels = [label]  # for nlp.pipe_labels
        self.merge = merge
        self.serialization_fields += [
            "pattern",
            "destructive",
            "label",
            "labels",
            "merge",
        ]

        if name:
            self.name = name
        else:
            self.name = "regex_ruler_" + label

    def __call__(self, doc: Doc) -> Doc:
        spans = []
        for m in self.pattern.finditer(doc.text):
            i, j = m.span()
            span = doc.char_span(i, j, label=self.label)
            if not span and self.destructive:
                destruct_token(doc, i, j)
                span = doc.char_span(i, j, label=self.label)
            if span:
                spans.append(span)
        doc.ents = merge_entities(doc.ents, tuple(spans))
        if self.merge:
            with doc.retokenize() as retokenizer:
                for span in spans:
                    retokenizer.merge(span)
        return doc


RE_POSTCODE = r"〒?(?<![\d-ー])\d{3}[\-ー]\d{4}(?![\d\-ー])"
LABEL_POSTCODE = "POSTCODE"
postcode_ruler = RegexRuler(label=LABEL_POSTCODE, pattern=RE_POSTCODE)

RE_CARCODE = r"\p{Han}+\s*\d+\s*\p{Hiragana}\s*\d{2,4}"
LABEL_CARCODE = "CARCODE"
carcode_ruler = RegexRuler(label=LABEL_CARCODE, pattern=RE_CARCODE)


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
        ents = []
        for m in self.REGEXP_WAREKI_YMD.finditer(doc.text):
            span = doc.char_span(*m.span(), label=self.LABEL)
            ents.append(span)

        for m in self.REGEXP_SEIREKI_YMD.finditer(doc.text):
            if self._is_valid_seireki(m.groups()):
                span = doc.char_span(*m.span(), label=self.LABEL)
                ents.append(span)
        doc.ents = merge_entities(doc.ents, ents)
        return doc

    @staticmethod
    def _is_valid_seireki(found_expressions: Tuple[str, str, str, str]) -> bool:
        year = int(found_expressions[0])
        month = int(found_expressions[1])
        day = int(found_expressions[2])
        return (1500 <= year <= 2100) and (1 <= month <= 12) and (1 <= day <= 31)


LABEL_DATE = DateRuler.LABEL

Language.factories[DateRuler.name] = DateRuler
