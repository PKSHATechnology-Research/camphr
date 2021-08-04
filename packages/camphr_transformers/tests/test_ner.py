from dataclasses import dataclass
from hypothesis import strategies as st, given
from pathlib import Path
from typing import Dict, List, Literal, Tuple, Union
from typing_extensions import TypeAlias
from camphr.serde import from_disk, to_disk
import pytest
from camphr_transformers.ner import Ner, _decode_bio, _DUMMY
from .utils import FIXTURE_DIR


T_TESTCASE: TypeAlias = Tuple[str, List[Tuple[str, str]]]


@dataclass
class ModelInfo:
    size: Literal["small", "large"]
    testcases: List[T_TESTCASE]


SMALL_MODEL = str(FIXTURE_DIR / "ner_bert")
TESTCASES_EN: List[T_TESTCASE] = [
    (
        "This is a test",
        [("a", "ORG")],
    ),
    (
        "暑熱厚",
        [],
    ),
]
TESTCASES_LARGE0: List[T_TESTCASE] = [
    (
        "Hugging Face Inc. is a company based in New York City.",
        [("Hugging Face Inc", "ORG"), ("New York City", "LOC")],
    ),
    (
        "Hugging Face Inc. 熱厚暑 is a company based in New York City.",
        [("Hugging Face Inc", "ORG"), ("New York City", "LOC")],
    ),
    (
        "a " * 1000,
        [],
    ),
]


@pytest.fixture(scope="session")
def nlp(request) -> Ner:
    return Ner(request.param)


SCENARIOS: Dict[str, ModelInfo] = {
    SMALL_MODEL: ModelInfo("small", TESTCASES_EN),
    "dslim/bert-base-NER": ModelInfo("large", TESTCASES_LARGE0),
}


PARAMS_RUN_NER: List[Tuple[str, str, List[Tuple[str, str]]]] = []
for k, info in SCENARIOS.items():
    for case in info.testcases:
        param = (k, *case)
        if info.size == "large":
            PARAMS_RUN_NER.append(pytest.param(*param, marks=pytest.mark.slow))  # type: ignore
        else:
            PARAMS_RUN_NER.append(param)


@pytest.mark.parametrize("nlp,text,expected", PARAMS_RUN_NER, indirect=["nlp"])
def test_ner(nlp: Ner, text: str, expected: List[Tuple[str, str]]):
    run_ner(nlp, text, expected)


@pytest.mark.parametrize("nlp", [SMALL_MODEL], indirect=["nlp"])
def test_long(nlp: Ner):
    text = "a " * 1000
    with pytest.raises(ValueError):
        nlp(text)


def run_ner(nlp: Ner, text: str, expected: List[Tuple[str, str]]):
    doc = nlp(text)
    assert doc.ents is not None
    ret = [(e.text, e.label) for e in doc.ents]
    assert ret == expected


PARAMS_SERDE: List[Tuple[str, T_TESTCASE]] = []
for k, info in SCENARIOS.items():
    param = (k, info.testcases[0])
    if info.size == "large":
        PARAMS_SERDE.append(pytest.param(*param, marks=pytest.mark.slow))  # type: ignore
    else:
        PARAMS_SERDE.append(param)


@pytest.mark.parametrize("nlp,case", PARAMS_SERDE, indirect=["nlp"])
def test_serde(nlp: Ner, case: T_TESTCASE, tmpdir: str):
    path = Path(tmpdir)
    to_disk(nlp, path)
    nlp2 = from_disk(path)
    assert isinstance(nlp2, Ner)
    run_ner(nlp2, *case)


@given(st.text())
@pytest.mark.parametrize("nlp", [SMALL_MODEL], indirect=True)
def test_fuzzy_ner(nlp: Ner, text: str):
    doc = nlp(text)
    assert doc.ents is not None
    assert doc.tokens is None


@pytest.mark.parametrize(
    ["text", "tokens", "labels", "mask", "expected"],
    [
        (
            "foo bar baz",
            ["foo", "bar", "baz"],
            ["O", "B-FOO", "I-FOO"],
            [0, 0, 0],
            [("bar baz", "FOO")],
        ),
        (
            "foo bar baz",
            ["foo", "bar", "baz"],
            ["O", "B-FOO", "I-BAR"],
            [0, 0, 0],
            [("bar", "FOO"), ("baz", "BAR")],
        ),
        (
            "foo bar baz",
            ["foo", "bar", "baz"],
            ["B-FOO", "I-BAR", "I-BAR"],
            [0, 0, 0],
            ["foo", "bar baz"],
        ),
        (
            "foo bar baz",
            ["foo", "[UNK]", "baz"],
            ["O", "B-FOO", "I-FOO"],
            [0, 1, 0],
            ["baz"],
        ),
        (
            "foo bar baz",
            ["foo", "[UNK]", "baz"],
            ["B-FOO", "I-FOO", "I-FOO"],
            [0, 1, 0],
            ["foo bar baz"],
        ),
        (
            "foo bar baz",
            ["foo", "[UNK]", "[UNK]", "baz"],
            ["B-FOO", "I-FOO", "I-FOO", "I-FOO"],
            [0, 1, 1, 0],
            ["foo bar baz"],
        ),
        (
            "foo bar baz",
            ["[CLS]", "foo", "[UNK]", "[UNK]", "baz", "[SEP]"],
            ["O", "B-FOO", "I-FOO", "I-FOO", "I-FOO", "I-FOO"],
            [1, 0, 1, 1, 0, 1],
            ["foo bar baz"],
        ),
        (
            "foo " + _DUMMY + "  baz",
            ["_DUMMY", "foo", "[UNK]", "[UNK]", "baz", "[SEP]"],
            ["O", "B-FOO", "I-FOO", "I-FOO", "I-FOO", "I-FOO"],
            [1, 0, 1, 1, 0, 1],
            ["foo " + _DUMMY + "  baz"],
        ),
    ],
)
def test_decode_bio(
    text: str,
    tokens: List[str],
    mask: List[int],
    labels: List[str],
    expected: Union[List[str], List[Tuple[str, str]]],
):
    ret = _decode_bio(text, tokens, mask, labels)
    assert ret.tokens is None
    assert ret.ents is not None

    a = [e.text for e in ret.ents]
    b = [(e.text, e.label) for e in ret.ents]
    assert a == expected or b == expected
