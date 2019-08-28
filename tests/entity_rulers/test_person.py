from bedoner import Japanese
from bedoner.entity_rulers import create_person_ruler
from bedoner.entity_rulers.labels import L
from spacy.tokens.doc import Doc
from spacy.tokens.span import Span
from collections import namedtuple


def test_person_entity_ruler():
    nlp = Japanese()
    nlp.add_pipe(create_person_ruler(nlp))

    Case = namedtuple("Case", ["text", "ent"])
    tests = [Case("今日は高松隆と海に行った", "高松隆"), Case("今日は田中と海に行った", "田中")]

    for case in tests:
        doc: Doc = nlp(case.text)
        assert len(doc.ents) == 1
        span: Span = doc.ents[0]
        assert span.text == case.ent
        assert span.label_ == L.PERSON
