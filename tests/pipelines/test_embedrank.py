import pytest
import torch
import spacy

from bedoner.models import trf_model
from bedoner.pipelines.embedrank import EMBEDRANK_KEYPHRASES, EmbedRank


@pytest.fixture(scope="module", params=["mecab"])
def nlp(request, trf_dir):
    lang = request.param
    _nlp = trf_model(lang, trf_dir)
    pipe = EmbedRank(vocab=_nlp.vocab)
    _nlp.add_pipe(pipe)
    return _nlp


TEXTS = ["今日はいい天気だった"]


@pytest.mark.parametrize("text", TEXTS)
def test_embedrank(nlp, text):
    doc = nlp(text)
    assert doc._.get(EMBEDRANK_KEYPHRASES) is not None


@pytest.mark.parametrize("text", TEXTS)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda test")
def test_embedrank_with_cuda(nlp, text):
    doc = nlp(text)
    assert doc._.get(EMBEDRANK_KEYPHRASES) is not None


def test_serialization(nlp, tmpdir):
    nlp.to_disk(str(tmpdir))
    nlp = spacy.load(str(tmpdir))
    text = "今日はいい天気だった"
    doc = nlp(text)
    assert doc._.get(EMBEDRANK_KEYPHRASES) is not None
