from pathlib import Path

import pytest
import spacy
import toml

from .utils import check_juman, check_knp, check_lang, check_serialization

with (Path(__file__).parent / "../pyproject.toml") as f:
    conf = toml.load(f)
LANGS = conf["tool"]["poetry"]["plugins"]["spacy_languages"]
PIPES = conf["tool"]["poetry"]["plugins"]["spacy_factories"]


@pytest.fixture(params=LANGS)
def lang(request):
    name = request.param
    if not check_lang(name):
        pytest.skip(f"{name} is required")
    return name


def test_blank(lang):
    spacy.blank(lang)


@pytest.fixture
def nlp():
    return spacy.blank("en")


SKIPS = {"knp": check_knp(), "juman_sentencizer": check_juman()}


@pytest.mark.parametrize("name", PIPES)
def test_pipe(nlp, name):
    if SKIPS.get(name, True):
        nlp.create_pipe(name)
        check_serialization(nlp)
    else:
        pytest.skip()
