from pathlib import Path

import pytest
import spacy
from spacy.vocab import Vocab

from camphr_core.lang.sentencepiece import EXTS, SentencePieceLang


@pytest.fixture
def nlp(spiece_path):
    return SentencePieceLang(Vocab(), meta={"tokenizer": {"model_path": spiece_path}})


TESTCASE = [
    ("    New York  ", ["New", "Y", "or", "k"]),
    ("今日はいい天気だった", ["今日は", "いい", "天気", "だった"]),
    (" 今日は\tいい天気　だった", ["今日は", "いい", "天気", "だった"]),
]


@pytest.mark.parametrize("text,tokens", TESTCASE)
def test_sentencepiece(nlp, text: str, tokens, spiece):
    doc = nlp(text)
    assert doc.text == text.replace("　", " ").replace("\t", " ").strip()
    for token, expected in zip(doc, tokens):
        assert token.text == expected

    # doc test
    pieces_ = spiece.encode_as_pieces(text)
    assert doc._.get(EXTS.pieces_) == pieces_


def test_serialize(nlp: SentencePieceLang, tmpdir: Path):
    nlp.to_disk(str(tmpdir))
    nlp2 = spacy.load(str(tmpdir))
    text, tokens = TESTCASE[0]
    doc = nlp2(text)
    assert doc.text == text.replace("　", " ").replace("\t", " ").strip()
    for token, expected in zip(doc, tokens):
        assert token.text == expected
