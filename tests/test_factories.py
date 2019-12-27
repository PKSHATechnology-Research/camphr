from pathlib import Path

import pytest
import spacy
import toml
from spacy.language import Language

with (Path(__file__).parent / "../pyproject.toml") as f:
    conf = toml.load(f)
LANGS = conf["tool"]["poetry"]["plugins"]["spacy_languages"]
PIPES = conf["tool"]["poetry"]["plugins"]["spacy_factories"]


@pytest.fixture(params=PIPES)
def pipe(request):
    return request.param


@pytest.fixture(params=LANGS)
def lang(request):
    return request.param


def test_foo(lang):
    spacy.blank(lang)


@pytest.fixture
def nlp():
    return spacy.blank("en")


def test_pipe(nlp: Language, pipe: str):
    nlp.create_pipe(pipe)
