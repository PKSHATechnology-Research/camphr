import pytest
from camphr.models import trf_model
from camphr.pipelines.trf_maskedlm import (
    PIPES,
    add_maskedlm_pipe,
    remove_maskedlm_pipe,
)
from camphr.torch_utils import get_loss_from_docs
from spacy.language import Language
from tests.utils import check_serialization


@pytest.fixture(scope="module")
def nlp(bert_dir, device):
    _nlp = trf_model("mecab", bert_dir)
    add_maskedlm_pipe(_nlp)
    _nlp.to(device)
    return _nlp


TEXTS = ["BERTを賢くするには，テキストの穴埋め問題を解かせます!", "賢くなった後は，各タスクでファインチューニングしましょう"]


@pytest.mark.parametrize("text", TEXTS)
def test_call(nlp, text):
    nlp(text)


def test_update(nlp):
    docs = [nlp.make_doc(text) for text in TEXTS]
    optim = nlp.resume_training()
    nlp.update(docs, [{} for _ in range(len(docs))], optim)
    loss = get_loss_from_docs(docs)
    assert loss.item() > 0


def test_pipe(nlp):
    list(nlp.pipe(TEXTS))


def test_update_for_long_seqence(nlp):
    text = "Foo " * 2000
    optim = nlp.resume_training()
    nlp.update([text], [{}], optim)


def test_serialization(nlp):
    check_serialization(nlp)


def test_remove_maskedlm(nlp: Language):
    remove_maskedlm_pipe(nlp)
    assert PIPES.bert_for_maskedlm not in nlp.pipe_names
    assert PIPES.bert_for_maskedlm_preprocessor not in nlp.pipe_names
    add_maskedlm_pipe(nlp)
