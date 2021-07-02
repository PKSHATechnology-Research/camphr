from typing import Collection, List

from hypothesis import given
from hypothesis import strategies as st
import pytest
from spacy.tokens import Doc
from spacy.vocab import Vocab

from camphr_core.utils import split_keepsep, zero_pad


@pytest.fixture(scope="session")
def vocab():
    return Vocab()


def _zero_pad(a: Collection[List[int]], pad_value: int = 0) -> List[List[int]]:
    """for test"""
    if len(a) == 0:
        return []
    max_length = max(len(el) for el in a)
    return [el + [pad_value] * (max_length - len(el)) for el in a]


@given(st.lists(st.lists(st.integers())))
def test_zero_pad(a):
    assert zero_pad(a) == _zero_pad(a)


@pytest.mark.parametrize(
    "text,sep,expected",
    [
        ("A.B.", ".", ["A.", "B."]),
        ("A.B", ".", ["A.", "B"]),
        ("AB", ".", ["AB"]),
        ("", ".", [""]),
        (".", ".", ["."]),
        ("A..B..", "..", ["A..", "B.."]),
        ("A..B.", "..", ["A..", "B."]),
        ("A..B", "..", ["A..", "B"]),
        ("AB", "..", ["AB"]),
        ("", "..", [""]),
        ("..", "..", [".."]),
    ],
)
def test_split_keepsep(text, sep, expected):
    assert split_keepsep(text, sep) == expected
