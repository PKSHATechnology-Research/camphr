from pathlib import Path

import pytest
import spacy
import toml

with (Path(__file__).parent / "../pyproject.toml") as f:
    conf = toml.load(f)
LANGS = conf["tool"]["poetry"]["plugins"]["spacy_languages"]


@pytest.fixture(params=LANGS)
def lang(request):
    return request.param


def test_foo(lang):
    spacy.blank(lang)


@pytest.fixture
def nlp():
    return spacy.blank("en")
