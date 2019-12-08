import pytest
import spacy
from bedoner.models import trf_model
from bedoner.pipelines.embedrank import EMBEDRANK_KEYPHRASES, EmbedRank


@pytest.fixture(scope="module", params=["mecab"])
def nlp(request, trf_dir, device):
    lang = request.param
    _nlp = trf_model(lang, trf_dir)
    pipe = EmbedRank(vocab=_nlp.vocab)
    _nlp.add_pipe(pipe)
    _nlp.to(device)
    return _nlp


TEXTS = ["今日はいい天気だった"]


@pytest.mark.parametrize("text", TEXTS)
def test_embedrank(nlp, text):
    doc = nlp(text)
    assert doc._.get(EMBEDRANK_KEYPHRASES) is not None


def test_serialization(nlp, tmpdir):
    nlp.to_disk(str(tmpdir))
    nlp = spacy.load(str(tmpdir))
    text = "今日はいい天気だった"
    doc = nlp(text)
    assert doc._.get(EMBEDRANK_KEYPHRASES) is not None
