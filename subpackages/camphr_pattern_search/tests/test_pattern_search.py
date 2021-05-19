import re
from typing import List

import pytest
import spacy
from spacy.language import Language
from spacy.tokens import Doc
from spacy.tokens.span import Span
from tests.utils import check_mecab

from camphr_pattern_search.pattern_search import PatternSearcher

KEYWORDS = ["今日", "は", "明日", "lower", "mouse", "foobar", "走る", "頭痛", "BC", "AB ABC"]


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
    ("I live in AB ABC", ["AB ABC"], ["AB ABC"], "en", None),
    ("今日はいい天気だ", ["今日", "は"], None, "ja_mecab", None),
    ("Mice is a plural form of mouse", ["mouse"], None, "en", None),
    ("foo-bar", ["foo-bar"], ["foobar"], "en", None),
    ("たくさん走った", ["走っ"], ["走る"], "ja_mecab", None),
    ("走れ", ["走れ"], ["走る"], "ja_mecab", None),
    ("頭痛", ["頭痛"], ["ズツウ"], "ja_mecab", ["matched"]),
]


@pytest.mark.parametrize("text,expected,orig,target,labels", TESTCASES)
def test_call(
    nlp: Language, text: str, expected: List[str], orig, target, lang, labels
):
    if lang != target:
        pytest.skip(f"target lang is '{target}', but actual lang is {lang}")
    doc: Doc = nlp(text)
    ents = [span.text for span in doc.ents]
    assert ents == expected
    if labels is not None:
        assert [span.label_ for span in doc.ents] == labels


def test_overlaps(nlp: Language, lang: str):
    if lang != "en":
        pytest.skip()

    def f(doc: Doc) -> Doc:
        doc.ents += (Span(doc, 3, 5, "foo"),)
        return doc

    nlp.add_pipe(f, before="PatternSearcher")

    doc = nlp("aa bb cc foobar xx ww sss")
    assert len(doc.ents) == 1
    ent = doc.ents[0]
    assert ent.text == "foobar xx"
    assert ent.label_ == "foo"

    doc = nlp("aa bb foo bar xx ww sss")
    assert len(doc.ents) == 1
    ent = doc.ents[0]
    assert ent.text == "foo bar"
    assert ent.label_ == "matched"

    doc = nlp("lower aa bb foo bar xx ww sss")
    assert len(doc.ents) == 2
    assert [e.text for e in doc.ents] == ["lower", "foo bar"]
    assert [e.label_ for e in doc.ents] == ["matched", "foo"]


def test_overlaps2():
    nlp = spacy.blank("en")
    keywords = [
        "a bc",
        "b",
    ]

    model = PatternSearcher.get_model_from_words(keywords)
    pipe = PatternSearcher(
        model,
        lower=True,
        lemma=True,
    )
    nlp.add_pipe(pipe)

    doc = nlp("a bc ")
    assert [s.text for s in doc.ents] == ["a bc"]
