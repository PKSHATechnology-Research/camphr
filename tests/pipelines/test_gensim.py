import pytest
import spacy

from bedoner.pipelines.gensim import set_word2vec_vectors


@pytest.fixture
def fixture_fasttext(fixture_dir):
    return fixture_dir / "gensim/fasttext_tiny.model"


@pytest.mark.parametrize("text", ["日本語の埋め込みベクトルを計算します"])
def test_set_word2vec_vectors(fixture_fasttext, text):
    nlp = spacy.blank("mecab")
    doc = nlp(text)
    assert len(doc.vector) == 0
    set_word2vec_vectors(nlp, fixture_fasttext)
    doc = nlp(text)
    assert len(doc.vector) != 0
