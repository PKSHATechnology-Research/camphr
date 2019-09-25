from typing import Tuple

import pytest
from bedoner.lang.knp import Japanese
from bedoner.pipelines.knp_ner import KnpEntityExtractor
from spacy.tokens import Doc


@pytest.fixture
def nlp():
    _nlp = Japanese()
    _nlp.add_pipe(KnpEntityExtractor(nlp))
    return _nlp


@pytest.mark.parametrize(
    "text,ents", [("今日はいい天気だったので山田太郎と散歩に行きました", [("今日", "DATE"), ("山田太郎", "PERSON")])]
)
def test_knp_entity_extractor(nlp: Japanese, text: str, ents: Tuple[str]):
    doc: Doc = nlp(text)
    assert len(doc.ents) == len(ents)
    for s, expected_ent in zip(doc.ents, ents):
        assert s.text == expected_ent[0]
        assert s.label_ == expected_ent[1]
