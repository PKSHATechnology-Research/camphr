import pytest
import sentencepiece as spm
import torch
from camphr.lang.juman import Japanese as Juman
from camphr.lang.mecab import Japanese as Mecab
from spacy.vocab import Vocab

from .utils import BERT_DIR, FIXTURE_DIR, XLNET_DIR, check_juman, check_mecab


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


@pytest.fixture(scope="session", params=["bert", "xlnet"])
def trf_name(request):
    return request.param


@pytest.fixture(scope="session")
def trf_dir(trf_name):
    if trf_name == "bert":
        return str(BERT_DIR)
    if trf_name == "xlnet":
        return str(XLNET_DIR)


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
        pytest.skip()
    return torch.device("cuda")


@pytest.fixture(scope="session", params=["mecab", "juman", "sentencepiece"])
def lang(request):
    return request.param


@pytest.fixture(
    scope="session",
    params=[
        str(BERT_DIR),
        "xlnet-base-cased",
        "bert-base-uncased",
        str(XLNET_DIR),
        "bert-base-japanese",
    ],
)
def trf_name_or_path(request):
    return request.param
