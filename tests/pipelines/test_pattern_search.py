import re
from pathlib import Path

import pytest
import spacy
from spacy.tokens import Doc

from camphr.pipelines.pattern_search import PATTERN_MATCH_AS, PatternSearcher
from tests.utils import check_mecab

KEYWORDS = ["今日", "は", "明日", "lower", "mouse", "foobar", "走る"]


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
        destructive=False,
        lower=True,
        lemma=True,
        normalizer=lambda x: re.sub(r"\W", "", x),
    )
    _nlp.add_pipe(pipe)
    return _nlp


TESTCASES = [
    ("今日はいい天気だ", ["今日", "は"], None, "ja_mecab"),
    ("Mice is a plural form of mouse", ["mouse"], None, "en"),
    ("foo-bar", ["foo-bar"], ["foobar"], "en"),
    ("たくさん走った", ["走っ"], ["走る"], "ja_mecab"),
    ("走れ", ["走れ"], ["走る"], "ja_mecab"),
]


@pytest.mark.parametrize("text,expected,orig,target", TESTCASES)
def test_call(nlp, text, expected, orig, target, lang):
    if lang != target:
        pytest.skip(f"target lang is '{target}', but actual lang is {lang}")
    doc: Doc = nlp(text)
    ents = [span.text for span in doc.ents]
    assert ents == expected
    orig = orig or expected
    assert [span._.get(PATTERN_MATCH_AS) for span in doc.ents] == orig


def test_serialization(nlp, tmpdir, lang):
    text, expected, _, target = TESTCASES[0]
    if lang != target:
        pytest.skip(f"target lang is '{target}', but actual lang is {lang}")
    path = Path(tmpdir)
    nlp.to_disk(path)
    nlp = spacy.load(path)
    doc = nlp(text)
    ents = [span.text for span in doc.ents]
    assert ents == expected, list(doc)
