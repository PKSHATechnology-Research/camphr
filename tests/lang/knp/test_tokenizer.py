import pytest
from ...utils import check_knp

pytestmark = pytest.mark.skipif(not check_knp(), reason="knp is not always necessary")

TOKENIZER_TESTS = [("日本語だよ", ["日本", "語", "だ", "よ"])]
TAG_TESTS = [("日本語だよ", ["名詞,地名", "名詞,普通名詞", "判定詞,*", "助詞,終助詞"])]
TEST_SPACE = ["今日は いい天気だ"]


@pytest.mark.parametrize("text,expected_tokens", TOKENIZER_TESTS)
def test_knp_tokenizer(knp_tokenizer, text, expected_tokens):
    tokens = [token.text for token in knp_tokenizer(text)]
    assert tokens == expected_tokens


@pytest.mark.parametrize("text,expected_tags", TAG_TESTS)
def test_knp_tokenizer_tags(knp_tokenizer, text, expected_tags):
    tags = [token.tag_ for token in knp_tokenizer(text)]
    assert tags == expected_tags


@pytest.mark.parametrize("text", TEST_SPACE)
def test_spaces(knp_tokenizer, text):
    doc = knp_tokenizer(text)
    assert doc.text == text
