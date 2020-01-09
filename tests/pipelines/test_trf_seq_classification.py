import random

import pytest
import spacy
from camphr.models import trf_seq_classification
from camphr.pipelines.trf_seq_classification import (
    TOP_LABEL,
    TOPK_LABELS,
    BertForSequenceClassification,
)
from camphr.pipelines.trf_utils import CONVERT_LABEL
from camphr.torch_utils import get_loss_from_docs
from spacy.language import Language
from spacy.tests.util import assert_docs_equal


@pytest.fixture(scope="module")
def labels():
    return ["one", "two", "three"]


@pytest.fixture(scope="module")
def nlp(trf_dir, labels, torch_lang, device):
    _nlp = trf_seq_classification(torch_lang, pretrained=trf_dir, labels=labels)
    _nlp.to(device)
    return _nlp


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


@pytest.fixture(scope="module")
def labels2():
    return ["o", "t"]


@pytest.fixture(scope="module")
def nlp2(trf_dir, labels2, device, torch_lang):
    _nlp = trf_seq_classification(torch_lang, pretrained=trf_dir, labels=labels2)
    _nlp.to(device)
    return _nlp


def test_user_hooks(nlp2, docs_golds):
    pipe = nlp2.pipeline[-1][1]
    pipe.add_user_hook(CONVERT_LABEL, lambda x: x[0])
    optim = nlp2.resume_training()
    nlp2.update(*zip(*docs_golds), optim)
