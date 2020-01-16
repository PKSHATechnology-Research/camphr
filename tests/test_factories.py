from pathlib import Path

import pytest
import spacy
import toml
import pytest
from .utils import check_juman, check_lang, check_mecab

with (Path(__file__).parent / "../pyproject.toml") as f:
    conf = toml.load(f)
LANGS = conf["tool"]["poetry"]["plugins"]["spacy_languages"]


@pytest.fixture(params=LANGS)
def lang(request):
    name = request.param
    if not check_lang(name):
        pytest.skip()
    return name


def test_foo(lang):
    spacy.blank(lang)


@pytest.fixture
def nlp():
    return spacy.blank("en")
