import random

import pytest
from spacy.language import Language

from camphr.models import create_model
from camphr.pipelines.transformers.model import TRANSFORMERS_MODEL
from camphr.pipelines.transformers.seq_classification import (
    TOP_LABEL,
    TOPK_LABELS,
    TRANSFORMERS_SEQ_CLASSIFIER,
    TrfForSequenceClassification,
)
from camphr.pipelines.transformers.tokenizer import TRANSFORMERS_TOKENIZER
from camphr.torch_utils import get_loss_from_docs
from tests.utils import check_serialization


@pytest.fixture(scope="module")
def labels():
    return ["one", "two", "three"]


@pytest.fixture(scope="module")
def nlp(trf_name_or_path, labels, lang, device):
    config = f"""
    lang:
        name: {lang}
        torch: true
        optimizer:
            class: torch.optim.SGD
            params:
                lr: 0.01
    pipeline:
        {TRANSFORMERS_TOKENIZER}:
            trf_name_or_path: {trf_name_or_path}
        {TRANSFORMERS_MODEL}:
            trf_name_or_path: {trf_name_or_path}
        {TRANSFORMERS_SEQ_CLASSIFIER}:
            trf_name_or_path: {trf_name_or_path}
            labels: {labels}
    """
    return create_model(config)


TEXTS = ["トランスフォーマーズを使ってテキスト分類をします", "うまくclassificationできるかな?"]


@pytest.fixture(scope="module")
def docs_golds(labels):
    res = []
    for text in TEXTS:
        res.append((text, {"cats": {random.choice(labels): 1}}))
    return res


@pytest.mark.parametrize("text", TEXTS)
def test_call(nlp, text, labels):
    doc = nlp(text)
    assert set(labels) == set(doc.cats)


@pytest.mark.parametrize("text", TEXTS)
def test_ext(nlp, text, labels):
    doc = nlp(text)
    assert doc._.get(TOP_LABEL)
    k = 2
    assert len(doc._.get(TOPK_LABELS)(k)) == k


def test_update(nlp, labels, docs_golds):
    optim = nlp.resume_training()
    texts, labels = zip(*docs_golds)
    docs = [nlp.make_doc(text) for text in texts]
    nlp.update(docs, labels, optim)
    assert get_loss_from_docs(docs)


@pytest.mark.slow
def test_update_convergence(nlp, labels, docs_golds):
    nlp.pipeline[-1][1]
    optim = nlp.resume_training()
    texts, labels = zip(*docs_golds)
    prev = 1e9
    for i in range(10):
        docs = [nlp.make_doc(text) for text in texts]
        nlp.update(docs, labels, optim)
        assert prev > get_loss_from_docs(docs).item()


def test_serialization(nlp):
    check_serialization(nlp)


@pytest.mark.xfail(reason="see https://github.com/explosion/spaCy/pull/4664")
def test_eval(nlp: Language, docs_golds):
    score = nlp.evaluate(docs_golds)
    assert score.textcat_per_cat


def test_weights(vocab, labels):
    weights = range(len(labels))
    weights_map = dict(zip(labels, weights))
    pipe = TrfForSequenceClassification(vocab, labels=labels, label_weights=weights_map)
    assert pipe.label_weights.sum() == sum(weights)
