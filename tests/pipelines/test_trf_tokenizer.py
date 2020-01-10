from pathlib import Path

import pytest
from camphr.pipelines.trf_tokenizer import TransformersTokenizer
from camphr.pipelines.trf_utils import ATTRS
from spacy.tokens import Doc
from spacy.vocab import Vocab

from ..utils import BERT_DIR, XLNET_DIR


@pytest.fixture(scope="session")
def trf_tokenizer(trf_name_or_path):
    return TransformersTokenizer.from_pretrained(Vocab(), trf_name_or_path)


@pytest.mark.parametrize(
    "model_name,tokens,spaces,trf_tokens,align",
    [
        (
            str(BERT_DIR),
            ["EXILE", "の", "ATSUSHI"],
            [False, False, False],
            ["[CLS]", "EXILE", "の", "ATS", "##US", "##HI", "[SEP]"],
            [[1], [2], [3, 4, 5]],
        ),
        (
            "xlnet-base-cased",
            ["I", "am", "XLNet's", "tokenizer", "."],
            [True, True, True, False, False],
            [
                "▁I",
                "▁am",
                "▁",
                "XL",
                "Net",
                "'",
                "s",
                "▁token",
                "izer",
                ".",
                "<sep>",
                "<cls>",
            ],
            [[0], [1], [3, 4, 5, 6], [7, 8], [9]],
        ),
        (
            "bert-base-uncased",
            ["I", "am", "BERT's", "tokenizer", "."],
            [True, True, True, False, False],
            ["[CLS]", "i", "am", "bert", "'", "s", "token", "##izer", ".", "[SEP]"],
            [[1], [2], [3, 4, 5], [6, 7], [8]],
        ),
        (
            str(XLNET_DIR),
            ["EXILE", "の", "ATSUSHI"],
            [False, False, False],
            ["▁", "EX", "I", "LE", "の", "ATS", "U", "SHI", "<sep>", "<cls>"],
            [[1, 2, 3], [4], [5, 6, 7]],
        ),
        (
            "bert-base-japanese",
            ["EXILE", "の", "ATSUSHI"],
            [False, False, False],
            ["[CLS]", "EXILE", "の", "ATS", "##US", "##HI", "[SEP]"],
            [[1], [2], [3, 4, 5]],
        ),
    ],
)
def test_tokenizer(
    model_name,
    trf_tokenizer,
    trf_name_or_path,
    tokens,
    spaces,
    trf_tokens,
    align,
    tmp_path: Path,
):
    if model_name != trf_name_or_path:
        pytest.skip()
    doc = Doc(Vocab(), tokens, spaces=spaces)

    def check():
        _doc = trf_tokenizer(doc)
        assert _doc._.get(ATTRS.tokens) == trf_tokens
        assert _doc._.get(ATTRS.align) == align

    check()

    # save and restore
    tmp_path.mkdir(exist_ok=True)
    tmp_path.mkdir(exist_ok=True)
    trf_tokenizer.to_disk(tmp_path)
    trf_tokenizer = TransformersTokenizer(Vocab())
    trf_tokenizer.from_disk(tmp_path)
    check()


def test_long_sequence(trf_tokenizer: TransformersTokenizer):
    length = 10000
    doc = Doc(Vocab(), ["a"] * length, spaces=[True] * length)
    doc = trf_tokenizer(doc)
    assert len(doc._.get(ATTRS.token_ids)) <= trf_tokenizer.model.max_len
