from pathlib import Path
from typing import Sized, cast

import pytest
from spacy.tokens import Doc
from spacy.vocab import Vocab

from camphr.models import create_model
from camphr.pipelines.transformers.tokenizer import TRANSFORMERS_TOKENIZER, TrfTokenizer
from camphr.pipelines.transformers.utils import ATTRS

from ...utils import BERT_DIR, XLNET_DIR


@pytest.fixture(scope="session")
def trf_tokenizer(trf_name_or_path):
    return TrfTokenizer.from_pretrained(Vocab(), trf_name_or_path)


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
        pytest.skip("model type mismatch")
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
    trf_tokenizer = TrfTokenizer(Vocab())
    trf_tokenizer.from_disk(tmp_path)
    check()


TEXTS = ["This is a teeeest text for multiple inputs.", "複数の文章を入力した時の挙動を　，テストします"]


@pytest.fixture
def nlp(trf_name_or_path):
    config = f"""
    lang:
        name: en
        torch: true
        optimizer:
            class: torch.optim.SGD
    pipeline:
        {TRANSFORMERS_TOKENIZER}:
          trf_name_or_path: {trf_name_or_path}
    """
    return create_model(config)


def test_pipe(nlp):
    docs = list(nlp.pipe(TEXTS))
    x = TrfTokenizer.get_transformers_input(docs)
    assert len(x.input_ids) == len(TEXTS)


def test_update(trf_tokenizer, nlp):
    docs = [nlp.make_doc(text) for text in TEXTS]
    trf_tokenizer.update(docs, [{} for _ in range(len(TEXTS))])
    assert len(docs) == len(
        cast(Sized, TrfTokenizer.get_transformers_input(docs).input_ids)
    )


def test_long_sequence(trf_tokenizer: TrfTokenizer):
    length = 10000
    doc = Doc(Vocab(), ["a"] * length, spaces=[True] * length)
    doc = trf_tokenizer(doc)
    x = TrfTokenizer.get_transformers_input([doc])
    assert len(x.input_ids) <= trf_tokenizer.model.max_len
