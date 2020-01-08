from pathlib import Path

import pytest
import spacy
from spacy.tokens import Doc

import camphr.lang.mecab as mecab
from camphr.pipelines.pattern_search import PatternSearcher

KEYWORDS = ["今日", "は", "明日"]


@pytest.fixture(scope="module")
def nlp():
    _nlp = mecab.Japanese()
    pipe = PatternSearcher.from_words(KEYWORDS)
    _nlp.add_pipe(pipe)
    return _nlp


TESTCASES = [("今日はいい天気だ", ["今日", "は"])]


@pytest.mark.parametrize("text,expected", TESTCASES)
def test_call(nlp, text, expected):
    doc: Doc = nlp(text)
    ents = [span.text for span in doc.ents]
    assert ents == expected


def test_serialization(nlp, tmpdir):
    path = Path(tmpdir)
    nlp.to_disk(path)
    nlp = spacy.load(path)
    text, expected = TESTCASES[0]
    doc = nlp(text)
    ents = [span.text for span in doc.ents]
    assert ents == expected
