import tempfile

import pytest
import spacy
from bedoner.lang.juman import Japanese as Juman
from ...utils import check_juman

TOKENIZER_TESTS = [("日本語だよ", ["日本", "語", "だ", "よ"])]
TOKENIZER_TESTS_DIFFICULT = [
    ("けんさくえんじんぐーぐる", ["けんさく", "えんじん", "ぐー", "ぐる"], ["けんさく", "えんじん", "ぐーぐる"])
]

TAG_TESTS = [("日本語だよ", ["名詞,地名", "名詞,普通名詞", "判定詞,*", "助詞,終助詞"])]
TEST_SPACE = ["今日は いい天気だ"]


@pytest.mark.skipif(not check_juman(), reason="Is juman necessary?")
@pytest.mark.parametrize("text,expected_tokens", TOKENIZER_TESTS)
def test_juman_tokenizer(juman_tokenizer, text, expected_tokens):
    tokens = [token.text for token in juman_tokenizer(text)]
    assert tokens == expected_tokens


@pytest.mark.skipif(not check_juman(), reason="Is juman necessary?")
@pytest.mark.parametrize("text,expected_tags", TAG_TESTS)
def test_juman_tokenizer_tags(juman_tokenizer, text, expected_tags):
    tags = [token.tag_ for token in juman_tokenizer(text)]
    assert tags == expected_tags


@pytest.mark.parametrize("text,expected_tokens", TOKENIZER_TESTS)
def test_jumanpp_tokenizer(jumanpp_tokenizer, text, expected_tokens):
    tokens = [token.text for token in jumanpp_tokenizer(text)]
    assert tokens == expected_tokens


@pytest.mark.parametrize("text,expected_tags", TAG_TESTS)
def test_jumanpp_tokenizer_tags(jumanpp_tokenizer, text, expected_tags):
    tags = [token.tag_ for token in jumanpp_tokenizer(text)]
    assert tags == expected_tags


@pytest.mark.skipif(not check_juman(), reason="Is juman necessary?")
@pytest.mark.parametrize("text,expected_tokens,foo", TOKENIZER_TESTS_DIFFICULT)
def test_juman_tokenizer_difficult(juman_tokenizer, text, expected_tokens, foo):
    tokens = [token.text for token in juman_tokenizer(text)]
    assert tokens == expected_tokens


@pytest.mark.parametrize("text,foo,expected_tokens", TOKENIZER_TESTS_DIFFICULT)
def test_jumanpp_tokenizer_difficult(jumanpp_tokenizer, text, expected_tokens, foo):
    tokens = [token.text for token in jumanpp_tokenizer(text)]
    assert tokens == expected_tokens


@pytest.mark.skipif(not check_juman(), reason="Is juman necessary?")
def test_serialization():
    def foo(x):
        return 2 * x

    nlp = Juman(
        meta={"tokenizer": {"juman_kwargs": {"jumanpp": False}, "preprocessor": foo}}
    )
    with tempfile.TemporaryDirectory() as tmpd:
        nlp.to_disk(tmpd)
        nlp2 = spacy.load(tmpd)
    assert nlp.tokenizer.preprocessor(1) == nlp2.tokenizer.preprocessor(1)
    assert nlp2.tokenizer.tokenizer.command == "juman"


@pytest.mark.skipif(not check_juman(), reason="Is juman necessary?")
@pytest.mark.parametrize("text", TEST_SPACE)
def test_spaces(juman_tokenizer, text):
    doc = juman_tokenizer(text)
    assert doc.text == text
