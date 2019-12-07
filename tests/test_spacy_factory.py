import pytest
import spacy


@pytest.mark.parametrize("lang", ["mecab"])
def test_spacy_factory(lang):
    spacy.blank(lang)
