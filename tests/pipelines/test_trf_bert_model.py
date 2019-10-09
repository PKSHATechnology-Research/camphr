import pytest
from bedoner.models import bert_model


@pytest.fixture
def nlp():
    return bert_model()


def test_forward(nlp):
    doc = nlp("今日はいい天気です")
    assert doc._.trf_last_hidden_state is not None
