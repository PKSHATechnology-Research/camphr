from typing import Tuple

import pytest
from spacy.tokens import Doc

import bedoner.lang.knp as knp
from bedoner.models import knp_ner

from ..utils import check_knp

pytestmark = pytest.mark.skipif(not check_knp(), reason="knp is not always necessary")


@pytest.fixture
def nlp():
    return knp_ner()


@pytest.mark.parametrize(
    "text,ents", [("今日はいい天気だったので山田太郎と散歩に行きました", [("今日", "DATE"), ("山田太郎", "PERSON")])]
)
def test_knp_entity_extractor(nlp: knp.Japanese, text: str, ents: Tuple[str]):
    doc: Doc = nlp(text)
    assert len(doc.ents) == len(ents)
    for s, expected_ent in zip(doc.ents, ents):
        assert s.text == expected_ent[0]
        assert s.label_ == expected_ent[1]
