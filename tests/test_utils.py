import pytest

from bedoner.utils import split_keepsep, zero_pad, get_doc_char_span
from spacy.tokens import Doc


def test_zero_pad():
    a = [[1, 2], [2, 3, 4]]
    b = [[1, 2, 0], [2, 3, 4]]
    assert b == zero_pad(a)


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
