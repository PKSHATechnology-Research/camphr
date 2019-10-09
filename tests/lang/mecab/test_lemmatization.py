"""Copied from Spacy"""
import pytest
from ...utils import check_mecab

pytestmark = pytest.mark.skipif(
    not check_mecab(), reason="mecab is not always necessary"
)


@pytest.mark.parametrize(
    "word,lemma",
    [("新しく", "新しい"), ("赤く", "赤い"), ("すごく", "すごい"), ("いただきました", "いただく"), ("なった", "なる")],
)
def test_mecab_lemmatizer_assigns(mecab_tokenizer, word, lemma):
    test_lemma = mecab_tokenizer(word)[0].lemma_
    assert test_lemma == lemma
