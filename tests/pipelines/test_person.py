import pytest
from spacy.tokens.doc import Doc
from spacy.tokens.span import Span

import camphr.ner_labels.labels_ontonotes as L
from camphr.pipelines.person_ner import create_person_ruler
from camphr_core.lang.mecab import Japanese
from tests.utils import check_mecab

TESTS = [("今日は高松隆と海に行った", "高松隆"), ("今日は田中と海に行った", "田中")]


@pytest.mark.parametrize("text,ent", TESTS)
@pytest.mark.skipif(not check_mecab(), reason="mecab is required")
def test_person_entity_ruler(text: str, ent: str):
    nlp = Japanese()
    nlp.add_pipe(create_person_ruler(nlp))

    doc: Doc = nlp(text)
    assert len(doc.ents) == 1
    span: Span = doc.ents[0]
    assert span.text == ent
    assert span.label_ == L.PERSON
