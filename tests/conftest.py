import os
from pathlib import Path

import pytest

#  from camphr.lang.juman import Japanese as Juman
import camphr.tokenizer.mecab as mecab

from .utils import FIXTURE_DIR, TRF_TESTMODEL_PATH, check_juman, check_lang, check_mecab


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture(scope="session")
def mecab_tokenizer() -> mecab.Tokenizer:
    if not check_mecab():
        pytest.skip("mecab is required")
    return mecab.Tokenizer()


@pytest.fixture(scope="session")
def juman_tokenizer():
    if not check_juman():
        pytest.skip()
    from camphr.tokenizer.juman import Tokenizer

    return Tokenizer()


@pytest.fixture(scope="session")
def spiece_path():
    return str(FIXTURE_DIR / "spiece.model")


@pytest.fixture
def chdir(tmp_path: Path):
    tmp_path.mkdir(exist_ok=True)
    cwd = os.getcwd()
    os.chdir(tmp_path)
    yield
    os.chdir(cwd)
