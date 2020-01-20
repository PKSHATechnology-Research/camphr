from pathlib import Path

import pytest
import spacy
from spacy.tokens import Doc

from camphr.pipelines.pattern_search import PatternSearcher
from tests.utils import check_mecab

KEYWORDS = ["今日", "は", "明日"]


@pytest.fixture(scope="module", params=["en", "ja_mecab"])
def nlp(request):
    lang: str = request.param
    if lang == "ja_mecab" and not check_mecab():
        pytest.skip()
    _nlp = spacy.blank(request.param)
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
