from pathlib import Path

import pytest
import spacy
from spacy.language import Language
from spacy.vocab import Vocab

from camphr_test.utils import check_sentencepiece

pytestmark = pytest.mark.skipif(
    not check_sentencepiece(), reason="sentencepiece is not installed"
)


@pytest.fixture
def nlp(spiece_path):
    from camphr_lang.sentencepiece import SentencePieceLang

    return SentencePieceLang(Vocab(), meta={"tokenizer": {"model_path": spiece_path}})


TESTCASE = [
    ("    New York  ", ["New", "Y", "or", "k"]),
    ("今日はいい天気だった", ["今日は", "いい", "天気", "だった"]),
    (" 今日は\tいい天気　だった", ["今日は", "いい", "天気", "だった"]),
]


@pytest.mark.parametrize("text,tokens", TESTCASE)
def test_sentencepiece(nlp, text: str, tokens, spiece):
    from camphr_lang.sentencepiece import EXTS

    doc = nlp(text)
    assert doc.text == text.replace("　", " ").replace("\t", " ").strip()
    for token, expected in zip(doc, tokens):
        assert token.text == expected

    # doc test
    pieces_ = spiece.encode_as_pieces(text)
    assert doc._.get(EXTS.pieces_) == pieces_


def test_serialize(nlp: Language, tmpdir: Path):
    nlp.to_disk(str(tmpdir))
    nlp2 = spacy.load(str(tmpdir))
    text, tokens = TESTCASE[0]
    doc = nlp2(text)
    assert doc.text == text.replace("　", " ").replace("\t", " ").strip()
    for token, expected in zip(doc, tokens):
        assert token.text == expected
