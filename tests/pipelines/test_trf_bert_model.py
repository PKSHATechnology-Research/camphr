import pytest
import torch

from bedoner.models import bert_model


@pytest.fixture(scope="module", params=["mecab", "juman"], ids=["mecab", "juman"])
def nlp(request):
    return bert_model(lang=request.param)


def test_forward(nlp):
    doc = nlp("今日はいい天気です")
    assert doc._.trf_last_hidden_state is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda test")
def test_forward_cuda(nlp):
    assert nlp.to(torch.device("cuda"))
    doc = nlp("今日はいい天気です")
    assert doc._.trf_last_hidden_state is not None
