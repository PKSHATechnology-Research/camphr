from bedoner.torch_utils import get_loss_from_docs
import random

import pytest
import spacy
import torch
from spacy.language import Language
from spacy.tests.util import assert_docs_equal

from bedoner.models import trf_seq_classification
from bedoner.pipelines.trf_seq_classification import BertForSequenceClassification


@pytest.fixture(scope="module")
def labels():
    return ["one", "two", "three"]


@pytest.fixture(scope="module", params=["mecab", "juman", "sentencepiece"])
def nlp(trf_dir, labels, request):
    lang = request.param
    return trf_seq_classification(lang, pretrained=trf_dir, labels=labels)


TEXTS = ["トランスフォーマーズを使ってテキスト分類をします", "うまくclassificationできるかな?"]
GOLDS = []


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda test")
@pytest.mark.parametrize("text", TEXTS)
def test_call_cuda(nlp, text, labels, cuda):
    nlp.to(cuda)
    doc = nlp(text)
    assert set(labels) == set(doc.cats)


def test_update(nlp, labels, docs_golds):
    pipe = nlp.pipeline[-1][1]
    params = pipe.model.parameters()
    before_sum = sum(p.sum().item() for p in params)
    optim = nlp.resume_training()
    texts, labels = zip(*docs_golds)
    nlp.update(texts, labels, optim)
    after_sum = sum(p.sum().item() for p in params)
    assert after_sum != before_sum


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


def test_serialization(nlp, tmp_path):
    path = tmp_path / "foo"
    path.mkdir()
    text = TEXTS[0]
    doc = nlp(text)
    nlp.to_disk(path)
    nlp = spacy.load(path)
    assert_docs_equal(doc, nlp(text))


@pytest.mark.xfail(reason="see https://github.com/explosion/spaCy/pull/4664")
def test_eval(nlp: Language, docs_golds):
    score = nlp.evaluate(docs_golds)
    assert score.textcat_per_cat


def test_weights(vocab, labels):
    weights = range(len(labels))
    weights_map = dict(zip(labels, weights))
    pipe = BertForSequenceClassification(
        vocab, labels=labels, label_weights=weights_map
    )
    assert pipe.label_weights.sum() == sum(weights)
