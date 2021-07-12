import random

import pytest
from hypothesis import given
from hypothesis import strategies as st

from camphr.tokenizer.juman import _SEPS, Tokenizer, _split_text_for_juman

from ...utils import check_juman

pytestmark = pytest.mark.skipif(
    not check_juman(), reason="juman is not always necessary"
)


@pytest.fixture(scope="session")
def nlp():
    return Tokenizer()


@pytest.mark.parametrize(
    "name,text",
    [
        ("0", ("一" * 1000) + "。"),
        ("1", ("一" * 500) + "。"),
        ("2", (("一" * 500) + "。") * 10),
        ("3", (("一" * 1000) + "。") * 10),
        ("4", ("一" * 500 + "お" * 500) + "。"),
    ],
)
def test_long_sentece(nlp, name, text):
    doc = nlp(text)
    assert text == doc.text


VOCAB = ["あ", "い", "今", "一"]


@given(st.integers(-1000, 1000), st.sampled_from((10, 100, 3000, 10000)), st.booleans())
def test__split_text_for_juman(seed, length, use_sep):
    random.seed(seed)
    vocab = VOCAB
    if use_sep:
        vocab = vocab + _SEPS
    text = "".join(random.choices(VOCAB, k=length))
    lines = list(_split_text_for_juman(text))
    assert all(len(line.encode("utf-8")) < 4097 for line in lines)
    assert "".join(lines) == text


@pytest.mark.parametrize(
    "text",
    [
        ("一" * 1000) + "。",
        ("一" * 500) + "。",
        (("一" * 500) + "。") * 10,
        (("一" * 1000) + "。") * 10,
    ],
)
def test__split_text_for_juman2(text):
    lines = list(_split_text_for_juman(text))
    assert all(len(line.encode("utf-8")) < 4097 for line in lines)
    assert "".join(lines) == text
