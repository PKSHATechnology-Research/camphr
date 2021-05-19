import random

from camphr_test.utils import check_serialization
from camphr_torch.utils import get_loss_from_docs
import pytest
from spacy.language import Language

from camphr_transformers.seq_classification import (
    TOPK_LABELS,
    TOP_LABEL,
    TRANSFORMERS_MULTILABEL_SEQ_CLASSIFIER,
    TRANSFORMERS_SEQ_CLASSIFIER,
    TrfForSequenceClassification,
)


@pytest.fixture(scope="module")
def labels():
    return ["one", "two", "three"]


@pytest.fixture(scope="module", params=["single", "multiple"])
def textcat_type(request):
    return request.param


@pytest.fixture(scope="module")
def nlp(trf_name_or_path, labels, lang, device, textcat_type):
    config = f"""
    lang:
        name: {lang}
        torch: true
        optimizer:
            class: torch.optim.SGD
            params:
                lr: 0.01
    pipeline:
        {TRANSFORMERS_SEQ_CLASSIFIER if textcat_type == "single" else TRANSFORMERS_MULTILABEL_SEQ_CLASSIFIER}:
            trf_name_or_path: {trf_name_or_path}
            labels: {labels}
    """
    return create_model(config)


@pytest.fixture(scope="module")
def texts(lang: str):
    if lang.startswith("ja"):
        return ["トランスフォーマーズを使ってテキスト分類をします", "うまくclassificationできるかな?"]
    elif lang == "en":
        return [
            "A test text for transformers' sequence classification",
            "What exact kind of architecture of neural networks do I need for a sequence binary/multiclass classification?",
        ]


@pytest.fixture(scope="module")
def docs_golds(labels, texts):
    res = []
    for text in texts:
        random.shuffle(labels)
        cats = {label: random.uniform(0.0, 1.0) for label in labels}
        res.append((text, {"cats": cats}))
    return res


def test_call(nlp, texts, labels, textcat_type):
    for text in texts:
        doc = nlp(text)
        assert set(labels) == set(doc.cats)
        if textcat_type == "single":
            assert abs(sum(doc.cats.values()) - 1.0) < 1e-5
        elif textcat_type == "multiple":
            assert abs(sum(doc.cats.values()) - 1.0) > 1e-5


def test_underscore(nlp, texts, labels):
    for text in texts:
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
