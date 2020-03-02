import random

import pytest
from hypothesis import given
from hypothesis import strategies as st

from camphr.lang.juman import _SEPS, Japanese, _split_text_for_juman


@pytest.fixture(scope="session")
def nlp():
    return Japanese()


def test_long_sentece(nlp):
    text = "一" * 10000
    nlp(text)


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
