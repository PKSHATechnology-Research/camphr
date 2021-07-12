from camphr.doc import Doc, DocProto, Ent, EntProto, Span, SpanProto, Token, TokenProto
from typing import Any, List
import hypothesis.strategies as st
from hypothesis import given
import pytest


@pytest.mark.parametrize("words", [["This ", "is ", "a   ", "pen", "."]])
def test_doc(words: List[str]):
    doc = Doc.from_words(words)
    assert doc.text == "".join(words)
    assert doc.tokens is not None
    for word, token in zip(words, doc.tokens):
        assert word == doc.text[token.start_char : token.end_char]


@given(st.lists(st.text()))
def test_doc_hyp(words: List[str]):
    doc = Doc.from_words(words)
    assert doc.text == "".join(words)
    assert doc.tokens is not None
    for word, token in zip(words, doc.tokens):
        assert word == doc.text[token.start_char : token.end_char]


doc = Doc("foo")
span = Span(0, 0, doc)
token = Token(0, 0, doc)
ent = Ent(0, 0, doc)


@pytest.mark.parametrize(
    "obj,ty,ok",
    [
        (doc, DocProto, True),
        (token, TokenProto, True),
        (token, SpanProto, True),
        (span, SpanProto, True),
        (ent, EntProto, True),
        (ent, SpanProto, True),
        (span, EntProto, False),
        (span, DocProto, False),
        (span, TokenProto, False),
    ],
)
def test_protocol(obj: Any, ty: Any, ok: bool):
    assert isinstance(obj, ty) == ok
