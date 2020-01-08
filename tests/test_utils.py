from typing import Iterable, List

import pytest
from camphr.utils import get_doc_char_span, split_keepsep, zero_pad
from hypothesis import given
from hypothesis import strategies as st
from spacy.tokens import Doc


def _zero_pad(a: Iterable[List[int]], pad_value: int = 0) -> List[List[int]]:
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


@pytest.mark.parametrize(
    "tokens,i,j,expected",
    [(["Foo", "bar", "baz"], 0, 2, "Fo"), (["Foo", "bar", "baz"], 0, 5, "Foo b")],
)
def test_get_doc_char_span(vocab, tokens, i, j, expected):
    doc = Doc(vocab, tokens)
    span = get_doc_char_span(doc, i, j)
    assert span.text == expected
