from pathlib import Path

from camphr_test.utils import check_lang
import pytest
import spacy
import toml

with (Path(__file__).parent / "../pyproject.toml") as f:
    conf = toml.load(f)

LANGS = conf["tool"]["poetry"]["plugins"]["spacy_languages"]


@pytest.fixture(params=LANGS)
def lang(request):
    name = request.param
    if not check_lang(name):
        pytest.skip(f"{name} is required")
    return name


def test_blank(lang):
    spacy.blank(lang)
