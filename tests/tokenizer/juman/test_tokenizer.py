from pathlib import Path
import tempfile

import pytest

from camphr.tokenizer.juman import Tokenizer

from ...utils import check_juman

pytestmark = pytest.mark.skipif(
    not check_juman(), reason="juman is not always necessary"
)

TOKENIZER_TESTS = [("日本語だよ", ["日本", "語", "だ", "よ"])]
TOKENIZER_TESTS_DIFFICULT = [
    ("けんさくえんじんぐーぐる", ["けんさく", "えんじん", "ぐー", "ぐる"], ["けんさく", "えんじん", "ぐーぐる"])
]

TAG_TESTS = [("日本語だよ", ["名詞,地名", "名詞,普通名詞", "判定詞,*", "助詞,終助詞"])]
TEST_SPACE = ["今日は\u3000\u3000いい天気だ"]


@pytest.mark.parametrize("text,expected_tokens", TOKENIZER_TESTS)
def test_juman_tokenizer(juman_tokenizer: Tokenizer, text: str, expected_tokens):
    tokens = [token.text for token in juman_tokenizer(text)]
    assert tokens == expected_tokens


@pytest.mark.parametrize("text,expected_tags", TAG_TESTS)
def test_juman_tokenizer_tags(juman_tokenizer: Tokenizer, text: str, expected_tags):
    tags = [token.tag_ for token in juman_tokenizer(text)]
    assert tags == expected_tags


@pytest.mark.parametrize("text,expected0,expected1", TOKENIZER_TESTS_DIFFICULT)
def test_juman_tokenizer_difficult(
    juman_tokenizer: Tokenizer, text: str, expected0, expected1
):
    tokens = [token.text for token in juman_tokenizer(text)]
    assert tokens == expected0 or tokens == expected1


def test_serialization():
    nlp = Tokenizer(juman_kwargs={"jumanpp": True})
    with tempfile.TemporaryDirectory() as tmpd:
        path = Path(tmpd)
        nlp.to_disk(path)
        nlp2 = Tokenizer.from_disk(path)
    assert nlp2.tokenizer.command == "jumanpp"


@pytest.mark.parametrize("text", TEST_SPACE)
def test_spaces(juman_tokenizer: Tokenizer, text: str):
    doc = juman_tokenizer(text)
    assert doc.text == text
