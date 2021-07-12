from typing import List, Tuple
import pytest

from ...utils import check_mecab

pytestmark = pytest.mark.skipif(
    not check_mecab(), reason="mecab is not always necessary"
)

TOKENIZER_TESTS: List[Tuple[str, List[str]]] = [
    ("日本語だよ", ["日本語", "だ", "よ"]),
    ("東京タワーの近くに住んでいます。", ["東京", "タワー", "の", "近く", "に", "住ん", "で", "い", "ます", "。"]),
    ("吾輩は猫である。", ["吾輩", "は", "猫", "で", "ある", "。"]),
    ("スペース に対応　できるかな？", ["スペース", "に", "対応", "できる", "か", "な", "？"]),
]

TEST_SPACE = ["今日は いい天気だ"]


TAG_TESTS: List[Tuple[str, List[str]]] = [
    ("日本語だよ", ["名詞,一般,*,*", "助動詞,*,*,*", "助詞,終助詞,*,*"]),
    (
        "東京タワーの近くに住んでいます。",
        [
            "名詞,固有名詞,地域,一般",
            "名詞,一般,*,*",
            "助詞,連体化,*,*",
            "名詞,副詞可能,*,*",
            "助詞,格助詞,一般,*",
            "動詞,自立,*,*",
            "助詞,接続助詞,*,*",
            "動詞,非自立,*,*",
            "助動詞,*,*,*",
            "記号,句点,*,*",
        ],
    ),
    (
        "吾輩は猫である。",
        [
            "名詞,代名詞,一般,*",
            "助詞,係助詞,*,*",
            "名詞,一般,*,*",
            "助動詞,*,*,*",
            "助動詞,*,*,*",
            "記号,句点,*,*",
        ],
    ),
    (
        "月に代わって、お仕置きよ!",
        [
            "名詞,一般,*,*",
            "助詞,格助詞,一般,*",
            "動詞,自立,*,*",
            "助詞,接続助詞,*,*",
            "記号,読点,*,*",
            "名詞,一般,*,*",
            "助詞,終助詞,*,*",
            "名詞,サ変接続,*,*",
        ],
    ),
    (
        "すもももももももものうち",
        [
            "名詞,一般,*,*",
            "助詞,係助詞,*,*",
            "名詞,一般,*,*",
            "助詞,係助詞,*,*",
            "名詞,一般,*,*",
            "助詞,連体化,*,*",
            "名詞,非自立,副詞可能,*",
        ],
    ),
]


@pytest.mark.parametrize("text,expected_tokens", TOKENIZER_TESTS)
def test_ja_tokenizer(mecab_tokenizer, text, expected_tokens):
    tokens = [token.text.strip() for token in mecab_tokenizer(text)]
    assert tokens == expected_tokens


@pytest.mark.parametrize("text", TEST_SPACE)
def test_spaces(mecab_tokenizer, text):
    doc = mecab_tokenizer(text)
    assert doc.text == text


@pytest.mark.parametrize("text,expected_tags", TAG_TESTS)
def test_ja_tokenizer_tags(mecab_tokenizer, text, expected_tags):
    tags = [token.tag_ for token in mecab_tokenizer(text)]
    assert tags == expected_tags


@pytest.mark.parametrize("word,lemma", [("foo", "foo"), ("今日", "今日"), ("走っ", "走る")])
def test_lemma(mecab_tokenizer, word, lemma):
    assert list(mecab_tokenizer("foo"))[0].lemma_ == "foo"
