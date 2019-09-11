import pytest
from bedoner.lang.mecab import Japanese
from bedoner.entity_rulers.date import DateRuler
from bedoner.ner_labels.labels_ontonotes import Labels as L
from spacy.tokens.doc import Doc
from spacy.tokens.span import Span


@pytest.fixture
def nlp():
    _nlp = Japanese()
    _nlp.add_pipe(DateRuler(_nlp))
    return _nlp


TESTS = [("今日は2019年11月30日だ", "2019年11月30日"), ("僕は平成元年4月10日生まれだ", "平成元年4月10日")]


@pytest.mark.parametrize("text,ent", TESTS)
def test_date_entity_ruler(nlp, text: str, ent: str):
    doc: Doc = nlp(text)
    assert len(doc.ents) == 1
    span: Span = doc.ents[0]
    assert span.text == ent
    assert span.label_ == L.DATE
