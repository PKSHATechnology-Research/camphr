from pathlib import Path

import pytest

FIXTURE_DIR = (Path(__file__).parent / "fixtures/").absolute()


@pytest.fixture(scope="session")
def spiece_path():
    return str(FIXTURE_DIR / "spiece.model")


@pytest.fixture(scope="session")
def spiece(spiece_path):
    import sentencepiece as spm

    s = spm.SentencePieceProcessor()
    s.load(spiece_path)
    return s
