import spacy
from pathlib import Path
import pytest
from bedoner.lang.sentencepiece import SentencePieceLang
from spacy.vocab import Vocab


@pytest.fixture
def nlp(fixture_dir):
    return SentencePieceLang(
        Vocab(), meta={"tokenizer": {"model_path": str(fixture_dir / "spiece.model")}}
    )


TESTCASE = [
    ("    New York  ", ["New", "Y", "or", "k"]),
    ("今日はいい天気だった", ["今日は", "いい", "天気", "だった"]),
    (" 今日は いい天気　だった", ["今日は", "いい", "天気", "だった"]),
    (" 今日は\tいい天気　だった", ["今日は", "いい", "天気", "だった"]),
]


@pytest.mark.parametrize("text,tokens", TESTCASE)
def test_sentencepiece(nlp, text: str, tokens):
    doc = nlp(text)
    assert doc.text == text.replace("　", " ").replace("\t", " ").strip()
    for token, expected in zip(doc, tokens):
        assert token.text == expected


def test_serialize(nlp: SentencePieceLang, tmpdir: Path):
    nlp.to_disk(str(tmpdir))
    nlp2 = spacy.load(str(tmpdir))
    text, tokens = TESTCASE[0]
    doc = nlp2(text)
    assert doc.text == text.replace("　", " ").replace("\t", " ").strip()
    for token, expected in zip(doc, tokens):
        assert token.text == expected
