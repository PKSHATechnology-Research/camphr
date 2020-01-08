from itertools import zip_longest

import pytest
from camphr.pipelines.regex_ruler import MultipleRegexRuler, RegexRuler
from spacy.tokens import Doc

from ..utils import check_mecab

pytestmark = pytest.mark.skipif(
    not check_mecab(), reason="mecab is not always necessary"
)


@pytest.mark.parametrize(
    "text,patterns,expected",
    [
        (
            "16日はいい天気だった",
            {"date": r"\d+日", "whether": "いい天気|晴れ"},
            [("16日", "date"), ("いい天気", "whether")],
        )
    ],
)
def test_multiple_regex_ruler(mecab_tokenizer, text, patterns, expected):
    doc: Doc = mecab_tokenizer(text)
    ruler = MultipleRegexRuler(patterns)
    doc = ruler(doc)
    for ent, (content, label) in zip_longest(doc.ents, expected):
        assert ent.text == content
        assert ent.label_ == label


def test_destructive(mecab_tokenizer):
    text = "今日はいい天気だ"
    doc: Doc = mecab_tokenizer(text)
    assert doc[0].text == "今日"
    ruler = RegexRuler("今", "今", True)
    doc = ruler(doc)
    assert len(doc.ents)
