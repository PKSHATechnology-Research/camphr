from typing import List, Tuple, Set
import pytest
from camphr_transformers.ner import Ner


@pytest.fixture(scope="session")
def nlp() -> Ner:
    return Ner("dslim/bert-base-NER")


@pytest.mark.parametrize(
    "text, expected",
    [
        (
            "My name is Wolfgang and I live in Berlin",
            {("Wolfgang", "PER"), ("Berlin", "LOC")},
        )
    ],
)
def test_ner(nlp: Ner, text: str, expected: Set[Tuple[str, str]]):
    doc = nlp(text)
    assert doc.ents is not None
    print(doc.ents)
    for e in doc.ents:
        entry = (e.text, e.label)
        assert entry in expected
