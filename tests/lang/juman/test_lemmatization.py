"""Copied from Spacy"""
import pytest

from ...utils import check_juman


@pytest.mark.parametrize(
    "word,lemma",
    [("新しく", "新しい"), ("赤く", "赤い"), ("すごく", "すごい"), ("いただきました", "いただく"), ("なった", "なる")],
)
@pytest.mark.skipif(not check_juman(), reason="Is juman necessary?")
def test_mecab_lemmatizer_assigns(juman_tokenizer, word, lemma):
    test_lemma = juman_tokenizer(word)[0].lemma_
    assert test_lemma == lemma
