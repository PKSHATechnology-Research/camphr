import os
from pathlib import Path

import yaml
import pytest
import sentencepiece as spm
import torch
from spacy.vocab import Vocab

from camphr.lang.juman import Japanese as Juman
from camphr.lang.mecab import Japanese as Mecab
from camphr.models import create_model
from camphr.pipelines.transformers.model import TRANSFORMERS_MODEL

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
def mecab_tokenizer():
    if not check_mecab():
        pytest.skip("mecab is required")
    return Mecab.Defaults.create_tokenizer()


@pytest.fixture(scope="session")
def juman_tokenizer(request):
    if not check_juman():
        pytest.skip()
    return Juman.Defaults.create_tokenizer(juman_kwargs={"jumanpp": True})


@pytest.fixture(scope="session")
def spiece_path():
    return str(FIXTURE_DIR / "spiece.model")


@pytest.fixture(scope="session")
def spiece(spiece_path):
    s = spm.SentencePieceProcessor()
    s.load(spiece_path)
    return s


@pytest.fixture(scope="session")
def vocab():
    return Vocab()


@pytest.fixture
def cuda():
    return torch.device("cuda")


@pytest.fixture(scope="session", params=["cuda", "cpu"])
def device(request):
    if request.param == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        pytest.skip("cuda is required")
    return torch.device("cuda")


ALL_LANGS = ["ja_mecab", "ja_juman"]


@pytest.fixture(scope="session", params=ALL_LANGS)
def lang(request):
    if not check_lang(request.param):
        pytest.skip(f"No requirements for {request.param}")
    return request.param


@pytest.fixture(scope="session", params=TRF_TESTMODEL_PATH)
def trf_name_or_path(request):
    name = request.param
    if "bert-base-japanese" in name and not check_mecab():
        pytest.skip("mecab is required")
    return name


@pytest.fixture(scope="session")
def trf_model_config(lang, trf_name_or_path, device):
    return yaml.safe_load(
        f"""
    lang:
        name: {lang}
        optimizer:
            class: torch.optim.SGD
            params:
                lr: 0.01
    pipeline:
        {TRANSFORMERS_MODEL}:
          trf_name_or_path: {trf_name_or_path}
    """
    )


@pytest.fixture(scope="module")
def nlp_trf_model(trf_model_config, device):
    _nlp = create_model(trf_model_config)
    _nlp.to(device)
    return _nlp


@pytest.fixture
def chdir(tmp_path: Path):
    tmp_path.mkdir(exist_ok=True)
    cwd = os.getcwd()
    os.chdir(tmp_path)
    yield
    os.chdir(cwd)
