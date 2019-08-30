import pytest
from bedoner.lang.mecab import Japanese


TOKENIZER_TESTS = [("日本語だよ", ["日本", "語", "だ", "よ"])]
TOKENIZER_TESTS_DIFFICULT = [
    ("けんさくえんじんぐーぐる", ["けんさく", "えんじん", "ぐー", "ぐる"], ["けんさく", "えんじん", "ぐーぐる"])
]

TAG_TESTS = [("日本語だよ", ["名詞/地名", "名詞/普通名詞", "判定詞/*", "助詞/終助詞"])]


@pytest.mark.parametrize("text,expected_tokens", TOKENIZER_TESTS)
def test_juman_tokenizer(juman_tokenizer, text, expected_tokens):
    tokens = [token.text for token in juman_tokenizer(text)]
    assert tokens == expected_tokens


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


@pytest.mark.parametrize("text,expected_tokens,foo", TOKENIZER_TESTS_DIFFICULT)
def test_juman_tokenizer_difficult(juman_tokenizer, text, expected_tokens, foo):
    tokens = [token.text for token in juman_tokenizer(text)]
    assert tokens == expected_tokens


@pytest.mark.parametrize("text,foo,expected_tokens", TOKENIZER_TESTS_DIFFICULT)
def test_jumanpp_tokenizer_difficult(jumanpp_tokenizer, text, expected_tokens, foo):
    tokens = [token.text for token in jumanpp_tokenizer(text)]
    assert tokens == expected_tokens
