from pathlib import Path

import pytest
import spacy
import toml

from camphr_test.utils import check_juman, check_knp, check_lang, check_serialization

with (Path(__file__).parent / "../pyproject.toml") as f:
    conf = toml.load(f)

PIPES = conf["tool"]["poetry"]["plugins"]["spacy_factories"]


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
