from itertools import chain
from pathlib import Path

import pytest
import sentencepiece as spm
import spacy
from spacy.vocab import Vocab

from camphr.lang.sentencepiece import EXTS, SentencePieceLang


@pytest.fixture
def nlp(spiece_path):
    return SentencePieceLang(Vocab(), meta={"tokenizer": {"model_path": spiece_path}})


TESTCASE = [
    ("    New York  ", ["New", "Y", "or", "k"], [[0, 1], [2], [3], [4]]),
    ("今日はいい天気だった", ["今日は", "いい", "天気", "だった"], [[0, 1], [2], [3], [4]]),
    (" 今日は\tいい天気　だった", ["今日は", "いい", "天気", "だった"], [[0, 1], [2, 3], [4], [5, 6]]),
]


@pytest.mark.parametrize("text,tokens,align", TESTCASE)
def test_sentencepiece(
    nlp, text: str, tokens, align, spiece: spm.SentencePieceProcessor
):
    doc = nlp(text)
    assert doc.text == text.replace("　", " ").replace("\t", " ").strip()
    for token, expected in zip(doc, tokens):
        assert token.text == expected

    # doc test
    pieces = spiece.encode_as_ids(text)
    pieces_ = spiece.encode_as_pieces(text)
    assert doc._.get(EXTS.pieces) == pieces
    assert doc._.get(EXTS.pieces_) == pieces_
    assert doc._.get(EXTS.alignment) == align
    assert len(doc) == len(align)

    # token test
    for i in range(len(doc)):
        assert doc[i]._.get(EXTS.pieces_) == [pieces_[j] for j in align[i]]
        assert doc[i]._.get(EXTS.pieces) == [pieces[j] for j in align[i]]
        assert doc[i]._.get(EXTS.alignment) == align[i]

    # span test
    for l in range(len(doc)):
        for r in range(l, len(doc)):
            assert doc[l:r]._.get(EXTS.alignment) == align[l:r]
            assert doc[l:r]._.get(EXTS.pieces) == [
                pieces[i] for i in chain.from_iterable(align[l:r])
            ]
            assert doc[l:r]._.get(EXTS.pieces_) == [
                pieces_[i] for i in chain.from_iterable(align[l:r])
            ]


def test_serialize(nlp: SentencePieceLang, tmpdir: Path):
    nlp.to_disk(str(tmpdir))
    nlp2 = spacy.load(str(tmpdir))
    text, tokens, align = TESTCASE[0]
    doc = nlp2(text)
    assert doc.text == text.replace("　", " ").replace("\t", " ").strip()
    assert doc._.get(EXTS.alignment) == align
    for token, expected in zip(doc, tokens):
        assert token.text == expected
