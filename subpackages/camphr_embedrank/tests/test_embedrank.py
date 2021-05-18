import pytest
import spacy
from spacy.language import Language

from camphr_embedrank.embedrank import EMBEDRANK_KEYPHRASES, EmbedRank


@pytest.fixture(scope="module")
def nlp():
    _nlp = spacy.blank("en")
    pipe = _nlp.create_pipe("tagger")
    pipe.begin_training()
    _nlp.add_pipe(pipe)
    EmbedRank.DefaultPatterns = {"keyword": [{"TEXT": "test"}]}  # hack for test
    pipe = EmbedRank(vocab=_nlp.vocab)
    _nlp.add_pipe(pipe)
    return _nlp


TEXTS = ["This is a test sentence."]


@pytest.mark.parametrize("text", TEXTS)
def test_embedrank(nlp: Language, text: str):
    doc = nlp(text)
    assert doc._.get(EMBEDRANK_KEYPHRASES) is not None


def test_serialization(nlp, tmp_path):
    nlp.to_disk(tmp_path)
    nlp = spacy.load(tmp_path)
    text = TEXTS[0]
    doc = nlp(text)
    assert doc._.get(EMBEDRANK_KEYPHRASES) is not None
