import re
from pathlib import Path
from typing import List

import pytest
import spacy
from spacy.language import Language
from spacy.tokens import Doc

from camphr.pipelines.pattern_search import PATTERN_MATCH_AS, PatternSearcher
from tests.utils import check_mecab

KEYWORDS = ["今日", "は", "明日", "lower", "mouse", "foobar", "走る", "頭痛"]


@pytest.fixture(scope="module", params=["en", "ja_mecab"])
def lang(request):
    return request.param


@pytest.fixture(scope="module")
def nlp(lang: str):
    if lang == "ja_mecab" and not check_mecab():
        pytest.skip()
    _nlp = spacy.blank(lang)
    model = PatternSearcher.get_model_from_words(KEYWORDS)
    pipe = PatternSearcher(
        model,
        lower=True,
        lemma=True,
        normalizer=lambda x: re.sub(r"\W", "", x.text),
    )
    _nlp.add_pipe(pipe)
    return _nlp


TESTCASES = [
    ("今日はいい天気だ", ["今日", "は"], None, "ja_mecab"),
    ("Mice is a plural form of mouse", ["mouse"], None, "en"),
    ("foo-bar", ["foo", "-", "bar"], ["foobar"], "en"),
    ("たくさん走った", ["走っ"], ["走る"], "ja_mecab"),
    ("走れ", ["走れ"], ["走る"], "ja_mecab"),
    ("頭痛", ["頭痛"], ["ズツウ"], "ja_mecab"),
]


@pytest.mark.parametrize("text,expected,orig,target", TESTCASES)
def test_call(nlp: Language, text: str, expected: List[str], orig, target, lang):
    if lang != target:
        pytest.skip(f"target lang is '{target}', but actual lang is {lang}")
    doc: Doc = nlp(text)
    ents = [span.text for span in doc.ents]
    assert ents == expected
