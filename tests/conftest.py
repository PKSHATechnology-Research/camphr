from bedoner.lang.trf_mixin import TransformersLanguageMixin
from bedoner.utils import inject_mixin
from pathlib import Path
import sentencepiece as spm

import pytest
from spacy.vocab import Vocab

from bedoner.lang.juman import Japanese as Juman
from bedoner.lang.knp import Japanese as KNP
from bedoner.lang.mecab import Japanese as Mecab
from bedoner.pipelines.wordpiecer import WordPiecer
from bedoner.pipelines.trf_model import XLNetModel

from .utils import check_juman, check_knp, check_mecab


@pytest.fixture(scope="session")
def mecab_tokenizer():
    if check_mecab():
        return Mecab.Defaults.create_tokenizer(dicdir="/usr/local/lib/mecab/dic/ipadic")


@pytest.fixture(scope="session")
def juman_tokenizer():
    if check_juman():
        return Juman.Defaults.create_tokenizer(juman_kwargs={"jumanpp": False})


@pytest.fixture(scope="session")
def jumanpp_tokenizer():
    return Juman.Defaults.create_tokenizer(juman_kwargs={"jumanpp": True})


@pytest.fixture(scope="session")
def knp_tokenizer():
    if check_knp():
        return KNP.Defaults.create_tokenizer()


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


@pytest.fixture(scope="session")
def xlnet_wp(xlnet_dir):
    return WordPiecer.from_pretrained(Vocab(), xlnet_dir)


@pytest.fixture(scope="session")
def xlnet_model(xlnet_dir):
    return XLNetModel.from_pretrained(Vocab(), xlnet_dir)


@pytest.fixture(scope="session")
def trf_mecab():
    return inject_mixin(TransformersLanguageMixin, Mecab)


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
