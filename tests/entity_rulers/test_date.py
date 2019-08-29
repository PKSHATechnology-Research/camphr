from bedoner.languages.mecab import Japanese
from bedoner.entity_rulers import date_ruler
import bedoner.ner_labels.labels_ontonotes as L
from spacy.tokens.doc import Doc
from spacy.tokens.span import Span
from collections import namedtuple


def test_person_entity_ruler():
    nlp = Japanese()
    nlp.add_pipe(date_ruler)

    Case = namedtuple("Case", ["text", "ent"])
    tests = [
        Case("今日は2019年11月30日だ", "2019年11月30日"),
        Case("僕は平成元年4月10日生まれだ", "平成元年4月10日"),
    ]
    for case in tests:
        doc: Doc = nlp(case.text)
        assert len(doc.ents) == 1
        span: Span = doc.ents[0]
        assert span.text == case.ent
        assert span.label_ == L.DATE
