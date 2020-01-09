import pytest
import spacy

LANGS = ["ja_mecab", "ja_juman", "ja_mecab_torch", "ja_juman_torch", "ja_mecab_torch"]


@pytest.mark.parametrize("lang", LANGS)
def test_load(lang: str):
    spacy.blank(lang)
