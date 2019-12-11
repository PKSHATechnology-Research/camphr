from pathlib import Path

import pytest
import sentencepiece as spm
import torch
from bedoner.lang.juman import Japanese as Juman
from bedoner.lang.mecab import Japanese as Mecab
from bedoner.pipelines.trf_model import XLNetModel
from bedoner.pipelines.wordpiecer import WordPiecer
from spacy.vocab import Vocab

from .utils import check_juman, check_mecab


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
    if check_mecab():
        return Mecab.Defaults.create_tokenizer(dicdir="/usr/local/lib/mecab/dic/ipadic")


@pytest.fixture(scope="session", params=[True, False])
def juman_tokenizer(request):
    if not check_juman():
        pytest.skip()
    return Juman.Defaults.create_tokenizer(juman_kwargs={"jumanpp": request.param})


@pytest.fixture(scope="session")
def DATADIR():
    return Path(__file__).parent / "data/"


@pytest.fixture(scope="session")
def fixture_dir():
    return (Path(__file__).parent / "fixtures/").absolute()


@pytest.fixture(scope="session")
def bert_dir(fixture_dir):
    return str(fixture_dir / "bert")


@pytest.fixture(scope="session")
def xlnet_dir(fixture_dir):
    return str(fixture_dir / "xlnet")


@pytest.fixture(scope="session", params=["bert", "xlnet"])
def trf_type(request):
    return request.param


@pytest.fixture(scope="session")
def pretrained(trf_type, fixture_dir):
    return str(fixture_dir / trf_type)


@pytest.fixture(scope="session")
def xlnet_wp(xlnet_dir):
    return WordPiecer.from_pretrained(Vocab(), xlnet_dir)


@pytest.fixture(scope="session")
def xlnet_model(xlnet_dir):
    return XLNetModel.from_pretrained(Vocab(), xlnet_dir)


@pytest.fixture(scope="session", params=["bert", "xlnet"])
def trf_name(request):
    return request.param


@pytest.fixture(scope="session")
def trf_dir(trf_name, bert_dir, xlnet_dir):
    if trf_name == "bert":
        return bert_dir
    if trf_name == "xlnet":
        return xlnet_dir


@pytest.fixture(scope="session")
def spiece_path(fixture_dir):
    return str(fixture_dir / "spiece.model")


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
        pytest.skip()
    return torch.device("cuda")


@pytest.fixture(scope="session", params=["mecab", "juman", "sentencepiece"])
def lang(request):
    return request.param
