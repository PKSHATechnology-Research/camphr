from hypothesis.strategies._internal.strategies import SearchStrategy
from camphr.doc import Doc, DocProto, T_Span
from camphr.utils import token_from_char_pos
from typing import Callable, List, Optional, Tuple, TypeVar

import pytest
from hypothesis import given
from hypothesis import strategies as st


@pytest.mark.parametrize(
    "tokens,i,expected",
    [
        (["foo ", "bar ", "baz "], 6, "bar "),
        (["foo ", "bar ", "baz "], 4, "bar "),
        (["foo ", "bar "], 8, None),
    ],
)
def test_get_doc_char_pos(tokens: List[str], i: int, expected: Optional[str]):
    doc = Doc.from_words(tokens)
    token = token_from_char_pos(doc, i)
    if expected is None:
        assert token is None
    else:
        token.text == expected


def _simple_get_doc_char_pos(doc: DocProto[T_Span], i: int) -> Optional[T_Span]:
    for token in doc:
        if token.l <= i < token.r:
            return token
    return None


T = TypeVar("T")


@st.composite
def tokens_and_char_idx(draw: Callable[[SearchStrategy[T]], T]):
    tokens = draw(st.lists(st.text()))
    l = len("".join(tokens))
    i = draw(st.integers(min_value=-l, max_value=2 * l))
    return tokens, i


@given(tokens_and_char_idx())
def test_get_doc_char_pos_hyp(data: Tuple[List[str], int]):
    tokens, i = data
    doc = Doc.from_words(tokens)
    ret = token_from_char_pos(doc, i)
    expected = _simple_get_doc_char_pos(doc, i)
    assert ret is expected
