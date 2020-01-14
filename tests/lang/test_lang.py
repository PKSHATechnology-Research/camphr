import pytest
import spacy

LANGS = ["ja_mecab", "ja_juman"]


@pytest.mark.parametrize("lang", LANGS)
def test_load(lang: str):
    spacy.blank(lang)
