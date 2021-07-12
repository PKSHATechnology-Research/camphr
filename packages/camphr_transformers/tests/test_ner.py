from pathlib import Path
from typing import Tuple, Set
from camphr.serde import from_disk, to_disk
import pytest
from camphr_transformers.ner import Ner
from .utils import FIXTURE_DIR


@pytest.fixture(scope="session")
def nlp() -> Ner:
    return Ner(str(FIXTURE_DIR / "ner_bert"))


TESTCASES_EN = [
    (
        "This is a test",
        {("This", "MISC"), ("is", "PER"), ("a", "ORG"), ("test", "PER")},
    )
]


@pytest.mark.parametrize("text, expected", TESTCASES_EN)
def test_ner(nlp: Ner, text: str, expected: Set[Tuple[str, str]]):
    run_ner(nlp, text, expected)


def run_ner(nlp: Ner, text: str, expected: Set[Tuple[str, str]]):
    doc = nlp(text)
    assert doc.ents is not None
    for e in doc.ents:
        print(e.text, e.label)
    for e in doc.ents:
        entry = (e.text, e.label)
        assert entry in expected


def test_serde(nlp: Ner, tmpdir: str):
    path = Path(tmpdir)
    to_disk(nlp, path)
    nlp2 = from_disk(path)
    assert isinstance(nlp2, Ner)
    run_ner(nlp2, *TESTCASES_EN[0])
