from pathlib import Path
from tests.utils import check_spm
from typing import List, TYPE_CHECKING, Tuple

import pytest


from camphr.tokenizer.sentencepiece import Tokenizer

if TYPE_CHECKING:
    import sentencepiece as spm

pytestmark = pytest.mark.skipif(
    not check_spm(), reason="sentencepiece is not always necessary"
)


@pytest.fixture
def nlp(spiece_path: str):
    return Tokenizer(model_path=spiece_path)


TESTCASE: List[Tuple[str, List[str]]] = [
    ("    New York  ", ["New", " Y", "or", "k"]),
    ("今日はいい天気だった", ["今日は", "いい", "天気", "だった"]),
    (" 今日は\tいい天気　だった", ["今日は", " ", "いい", "天気", " ", "だった"]),
]


@pytest.mark.parametrize("text,tokens", TESTCASE)
def test_sentencepiece(
    nlp: Tokenizer, text: str, tokens: List[str], spiece: "spm.SentencePieceProcessor"
):
    doc = nlp(text)
    assert doc.text.strip() == text.replace("　", " ").replace("\t", " ").strip()
    for token, expected in zip(doc, tokens):
        assert token.text == expected

    pieces_: List[str] = spiece.encode_as_pieces(text)  # type: ignore
    assert Tokenizer.get_spm_pieces(doc) == pieces_


def test_serialize(nlp: Tokenizer, tmpdir: Path):
    tmpdir = Path(tmpdir)
    nlp.to_disk(tmpdir)
    nlp2 = Tokenizer.from_disk(tmpdir)
    text, tokens = TESTCASE[0]
    doc = nlp2(text)
    assert doc.text == text.replace("　", " ").replace("\t", " ").strip()
    for token, expected in zip(doc, tokens):
        assert token.text == expected
