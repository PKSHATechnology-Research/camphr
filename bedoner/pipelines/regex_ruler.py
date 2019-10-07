import regex as re
from bedoner.utils import SerializationMixin, destruct_token
from spacy.tokens import Doc


class RegexRuler(SerializationMixin):
    def __init__(
        self, pattern, label: str, destructive: bool = False, merge: bool = False
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
        doc.ents += tuple(spans)
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
