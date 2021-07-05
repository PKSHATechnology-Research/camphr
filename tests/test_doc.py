from camphr.doc import Doc
from typing import List
import hypothesis.strategies as st
from hypothesis import given
import pytest


@pytest.mark.parametrize("words", [["This ", "is ", "a   ", "pen", "."]])
def test_doc(words: List[str]):
    doc = Doc.from_words(words)
    assert doc.text == "".join(words)
    assert doc.tokens is not None
    for word, token in zip(words, doc.tokens):
        assert word == doc.text[token.l : token.r]
