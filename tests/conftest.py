import pytest
from bedoner.lang.juman import Japanese as Juman
from bedoner.lang.knp import Japanese as KNP
from bedoner.lang.mecab import Japanese as Mecab
from bedoner.models import bert_wordpiecer


@pytest.fixture(scope="session")
def mecab_tokenizer():
    return Mecab.Defaults.create_tokenizer(dicdir="/usr/local/lib/mecab/dic/ipadic")


@pytest.fixture(scope="session")
def juman_tokenizer():
    return Juman.Defaults.create_tokenizer(juman_kwargs={"jumanpp": False})


@pytest.fixture(scope="session")
def jumanpp_tokenizer():
    return Juman.Defaults.create_tokenizer(juman_kwargs={"jumanpp": True})


@pytest.fixture(scope="session")
def knp_tokenizer():
    return KNP.Defaults.create_tokenizer()


@pytest.fixture(scope="session")
def bert_wordpiece_nlp():
    return bert_wordpiecer()
