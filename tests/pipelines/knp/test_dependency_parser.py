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
