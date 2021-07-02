from pathlib import Path

import pytest
from camphr_test.utils import check_juman, check_mecab
from camphr_lang.juman import Japanese as Juman
from camphr_lang.mecab import Japanese as Mecab

FIXTURE_DIR = (Path(__file__).parent / "fixtures/").absolute()


@pytest.fixture(scope="session", params=[True, False])
def juman_tokenizer(request):
    if not check_juman():
        pytest.skip()
    return Juman.Defaults.create_tokenizer(juman_kwargs={"jumanpp": request.param})


@pytest.fixture(scope="session")
def spiece_path():
    return str(FIXTURE_DIR / "spiece.model")


@pytest.fixture(scope="session")
def spiece(spiece_path):
    import sentencepiece as spm

    s = spm.SentencePieceProcessor()
    s.load(spiece_path)
    return s


@pytest.fixture(scope="session")
def mecab_tokenizer():
    if not check_mecab():
        pytest.skip("mecab is required")
    return Mecab.Defaults.create_tokenizer()
