import spacy
from pathlib import Path
import sentencepiece as spm
import pytest
from bedoner.lang.sentencepiece import SentencePieceLang, EXTS
from spacy.vocab import Vocab


@pytest.fixture
def nlp(spiece_path):
    return SentencePieceLang(Vocab(), meta={"tokenizer": {"model_path": spiece_path}})


TESTCASE = [
    ("    New York  ", ["New", "Y", "or", "k"], [1, 2, 3, 4]),
    ("今日はいい天気だった", ["今日は", "いい", "天気", "だった"], [1, 2, 3, 4]),
    (" 今日は\tいい天気　だった", ["今日は", "いい", "天気", "だった"], [1, 3, 4, 6]),
]


@pytest.mark.parametrize("text,tokens,align", TESTCASE)
def test_sentencepiece(
    nlp, text: str, tokens, align, spiece: spm.SentencePieceProcessor
):
    doc = nlp(text)
    assert doc.text == text.replace("　", " ").replace("\t", " ").strip()
    for token, expected in zip(doc, tokens):
        assert token.text == expected
    assert doc._.get(EXTS.pieces) == spiece.encode_as_ids(text)
    assert doc._.get(EXTS.pieces_) == spiece.encode_as_pieces(text)
    assert doc._.get(EXTS.alignment) == align


def test_serialize(nlp: SentencePieceLang, tmpdir: Path):
    nlp.to_disk(str(tmpdir))
    nlp2 = spacy.load(str(tmpdir))
    text, tokens, align = TESTCASE[0]
    doc = nlp2(text)
    assert doc.text == text.replace("　", " ").replace("\t", " ").strip()
    assert doc._.get(EXTS.alignment) == align
    for token, expected in zip(doc, tokens):
        assert token.text == expected
