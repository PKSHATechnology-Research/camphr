from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Tuple, Set
from camphr.serde import from_disk, to_disk
import pytest
from camphr_transformers.ner import Ner
from .utils import FIXTURE_DIR


SMALL_MODEL = str(FIXTURE_DIR / "ner_bert")
TESTCASES_EN = [
    (
        "This is a test",
        [("a", "ORG")],
    ),
]


@dataclass
class ModelInfo:
    size: Literal["small", "large"]
    testcases: List[Tuple[str, List[Tuple[str, str]]]]


@pytest.fixture(scope="session")
def nlp(request) -> Ner:
    return Ner(request.param)


SCENARIOS: Dict[str, ModelInfo] = {
    SMALL_MODEL: ModelInfo("small", TESTCASES_EN),
    "dslim/bert-base-NER": ModelInfo("large", TESTCASES_EN),
}


SCENARIOS_RUN_NER: List[Tuple[str, str, List[Tuple[str, str]]]] = []
for k, info in SCENARIOS.items():
    for case in info.testcases:
        param = (k, *case)
        if info.size == "large":
            SCENARIOS_RUN_NER.append(pytest.param(*param, marks=pytest.mark.slow))
        else:
            SCENARIOS_RUN_NER.append(param)


@pytest.mark.parametrize("nlp,text,expected", SCENARIOS_RUN_NER, indirect=["nlp"])
def test_ner(nlp: Ner, text: str, expected: List[Tuple[str, str]]):
    run_ner(nlp, text, expected)


def run_ner(nlp: Ner, text: str, expected: List[Tuple[str, str]]):
    doc = nlp(text)
    assert doc.ents is not None
    ret = [(e.text, e.label) for e in doc.ents]
    assert ret == expected


@pytest.mark.parametrize("nlp", [SMALL_MODEL], indirect=True)
def test_serde(nlp: Ner, tmpdir: str):
    path = Path(tmpdir)
    to_disk(nlp, path)
    nlp2 = from_disk(path)
    assert isinstance(nlp2, Ner)
    run_ner(nlp2, *TESTCASES_EN[0])
