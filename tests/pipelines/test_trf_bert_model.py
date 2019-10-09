import torch
import pytest
from bedoner.models import bert_model
from bedoner.torch_utils import TorchPipe
from bedoner.pipelines.trf_model import BertModel


@pytest.fixture
def nlp():
    return bert_model()


def test_forward(nlp):
    doc = nlp("今日はいい天気です")
    assert doc._.trf_last_hidden_state is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda test")
def test_forward_cuda(nlp):
    assert nlp.to(torch.device("cuda"))
    doc = nlp("今日はいい天気です")
    assert doc._.trf_last_hidden_state is not None
