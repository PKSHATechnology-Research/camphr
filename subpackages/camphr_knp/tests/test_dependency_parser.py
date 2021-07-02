import itertools

import pytest

from camphr.models import load

from ...utils import check_knp

pytestmark = pytest.mark.skipif(not check_knp(), reason="knp is not always necessary")


@pytest.fixture
def nlp():
    return load("knp")


@pytest.mark.parametrize("text,heads", [("太郎が本を読む", [4, 0, 4, 2, 4])])
def test_dependency_parse(nlp, text, heads):
    doc = nlp(text)
    for token, headi in itertools.zip_longest(doc, heads):
        assert token.head.i == headi


@pytest.mark.parametrize(
    "text,deps", [("太郎が本を読む", ["nsubj", "case", "obj", "case", "ROOT"])]
)
def test_dependency_deps(nlp, text, deps):
    doc = nlp(text)
    for token, depi in itertools.zip_longest(doc, deps):
        assert token.dep_ == depi


@pytest.mark.parametrize("text,heads,deps", [
    ("象は鼻が長い", [4, 0, 4, 2, 4], ["dislocated", "case", "nsubj", "case", "ROOT"]),
    ("リンゴとバナナとミカン", [0, 0, 0, 2, 0], ["ROOT", "case", "conj", "case", "conj"]),
    ("三匹の豚", [3, 0, 0, 3], ["nummod", "clf", "case", "ROOT"]),
    ("御盃を相交わす", [1, 4, 1, 4, 4], ["compound", "obj", "case", "advmod", "ROOT"]),
    ("子供を「僕」と呼ぶ", [6, 0, 3, 6, 3, 3, 6], ["obj", "case", "punct", "obl", "punct", "case", "ROOT"]),
    ("地震で半壊した家屋", [2, 0, 4, 2, 4], ["obl", "case", "acl", "aux", "ROOT"]),
    ("超重くて不完全", [1, 3, 3, 3], ["advmod", "advcl", "advmod", "ROOT"])
])
def test_dependency_parse_deps(nlp, text, heads, deps):
    doc = nlp(text)
    for token, headi, depi in itertools.zip_longest(doc, heads, deps):
        assert token.head.i == headi
        assert token.dep_ == depi
