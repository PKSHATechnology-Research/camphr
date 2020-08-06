import re
from pathlib import Path

import pytest
import spacy
from spacy.tokens import Doc

from camphr.pipelines.pattern_search import PatternSearcher
from tests.utils import check_mecab

KEYWORDS = ["今日", "は", "明日", "lower", "mouse", "foobar"]


@pytest.fixture(scope="module", params=["en", "ja_mecab"])
def lang(request):
    return request.param


@pytest.fixture(scope="module")
def nlp(lang):
    if lang == "ja_mecab" and not check_mecab():
        pytest.skip()
    _nlp = spacy.blank(lang)
    pipe = PatternSearcher.from_words(
        KEYWORDS,
        destructive=True,
        lower=True,
        lemma=True,
        normalizer=lambda x: re.sub(r"\W", "", x),
    )
    _nlp.add_pipe(pipe)
    return _nlp


TESTCASES = [
    ("今日はいい天気だ", ["今日", "は"], "ja_mecab"),
    ("Mice is a plural form of mouse", ["mouse"], "en"),
    ("foo-bar", ["foo-bar"], "en"),
]


@pytest.mark.parametrize("text,expected, target", TESTCASES)
def test_call(nlp, text, expected, target, lang):
    if lang != target:
        pytest.skip(f"target lang is '{target}', but actual lang is {lang}")
    doc: Doc = nlp(text)
    ents = [span.text for span in doc.ents]
    assert ents == expected


def test_serialization(nlp, tmpdir):
    path = Path(tmpdir)
    nlp.to_disk(path)
    nlp = spacy.load(path)
    text, expected, _ = TESTCASES[0]
    doc = nlp(text)
    ents = [span.text for span in doc.ents]
    assert ents == expected
