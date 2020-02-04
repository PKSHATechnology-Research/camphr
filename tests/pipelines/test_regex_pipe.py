from itertools import zip_longest

import pytest
import spacy
from spacy.language import Language
from spacy.tokens import Doc

from camphr.pipelines.regex_ruler import MultipleRegexRuler, RegexRuler

from ..utils import check_mecab, check_serialization

pytestmark = pytest.mark.skipif(
    not check_mecab(), reason="mecab is not always necessary"
)


TESTCASES = [
    (
        "16日はいい天気だった",
        {"date": r"\d+日", "whether": "いい天気|晴れ"},
        [("16日", "date"), ("いい天気", "whether")],
    )
]


@pytest.mark.parametrize("text,patterns,expected", TESTCASES)
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


@pytest.fixture(params=[("en", True), ("ja_mecab", check_mecab())])
def nlp(request):
    lang, not_skip = request.param
    if not not_skip:
        pytest.skip()
        return
    return spacy.blank(lang)


@pytest.mark.parametrize("text,patterns,expected", TESTCASES)
def test_serialization(nlp: Language, text, patterns, expected):
    pipe = MultipleRegexRuler(patterns)
    nlp.add_pipe(pipe)
    check_serialization(nlp)


@pytest.mark.parametrize(
    "text,pattern,expected",
    [("my phone number is 120-3141-2345", r"[\d-]+", ["120-3141-2345"])],
)
def test_rege_pipe(nlp: Language, text, pattern, expected):
    pipe = RegexRuler(pattern, "")
    nlp.add_pipe(pipe)
    doc = nlp(text)
    for ent, e in zip(doc.ents, expected):
        assert ent.text == e
