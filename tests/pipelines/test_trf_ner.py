import json
import tempfile

import pytest
import spacy
import torch
from spacy.gold import GoldParse
from spacy.language import Language

from bedoner.models import bert_ner
from bedoner.ner_labels.labels_ene import ALL_LABELS as enes
from bedoner.ner_labels.labels_irex import ALL_LABELS as irexs
from bedoner.ner_labels.utils import make_biluo_labels

from ..utils import in_ci


@pytest.fixture(scope="module")
def labels():
    return make_biluo_labels(enes)


@pytest.fixture(scope="module", params=["mecab", "juman"], ids=["mecab", "juman"])
def nlp(labels, request):
    lang = request.param
    _nlp = bert_ner(lang=lang, labels=["-"] + labels)
    assert _nlp.meta["lang"] == lang
    return _nlp


TESTCASE = [
    (
        "ＥＸＩＬＥのＡＴＳＵＳＨＩと中島美嘉が１４日ニューヨーク入り",
        {
            "entities": [
                (0, 5, "SHOW_ORGANIZATION"),
                (6, 13, "PERSON"),
                (14, 18, "PERSON"),
                (19, 22, "DATE"),
                (22, 28, "CITY"),
            ]
        },
    ),
    (
        "夏休み真っただ中の8月26日の夕方。",
        {"entities": [(0, 3, "DATE"), (9, 14, "DATE"), (15, 17, "TIME")]},
    ),
    ("。", {"entities": []}),
    (" おはよう", {"entities": []}),
    ("　おはよう", {"entities": []}),
]


@pytest.mark.parametrize("text,gold", TESTCASE)
def test_call(nlp: Language, text, gold):
    nlp(text)


def test_pipe(nlp: Language):
    list(nlp.pipe(["今日はいい天気なので外で遊びたい", "明日は晴れ"]))


@pytest.mark.parametrize("text,gold", TESTCASE)
def test_update(nlp: Language, text, gold):
    assert nlp.device.type == "cpu"
    doc = nlp(text)
    gold = GoldParse(doc, **gold)

    optim = nlp.resume_training()
    assert nlp.device.type == "cpu"
    doc = nlp.make_doc(text)
    assert doc._.loss is None
    nlp.update([doc], [gold], optim)
    assert doc._.loss


def test_update_batch(nlp: Language):
    texts, golds = zip(*TESTCASE)
    optim = nlp.resume_training()
    nlp.update(texts, golds, optim)


def test_evaluate(nlp: Language):
    nlp.evaluate(TESTCASE)


@pytest.mark.skipif(in_ci(), reason="Fail in circleci due to memory allocation error")
def test_save_and_load(nlp: Language):
    with tempfile.TemporaryDirectory() as d:
        nlp.to_disk(d)
        nlp = spacy.load(d)
        nlp(TESTCASE[0][0])


@pytest.fixture
def nlp_irex():
    return bert_ner(labels=["-"] + make_biluo_labels(irexs))


TESTCASE2 = ["資生堂の香水-禅とオードパルファンＺＥＮの違いを教えて下さい。また今でも製造されてますか？"]


@pytest.mark.parametrize("text", TESTCASE2)
def test_irex_call(nlp_irex: Language, text):
    nlp_irex(text)


def test_pipe_irex(nlp_irex: Language):
    list(nlp_irex.pipe(["今日はいい天気なので外で遊びたい", "明日は晴れ"]))


@pytest.fixture
def cuda():
    return torch.device("cuda")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda test")
@pytest.mark.parametrize("text,gold", TESTCASE)
def test_call_cuda(nlp: Language, text, gold, cuda):
    nlp.to(cuda)
    nlp(text)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda test")
@pytest.mark.parametrize("text,gold", TESTCASE)
def test_update_cuda(nlp: Language, text, gold, cuda):
    nlp.to(cuda)
    doc = nlp(text)
    gold = GoldParse(doc, **gold)

    optim = nlp.resume_training()
    doc = nlp.make_doc(text)
    assert doc._.loss is None
    nlp.update([doc], [gold], optim)
    assert doc._.loss


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda test")
def test_update_batch_cuda(nlp: Language, cuda):
    nlp.to(cuda)
    texts, golds = zip(*TESTCASE)
    optim = nlp.resume_training()
    nlp.update(texts, golds, optim)


@pytest.fixture(scope="module", params=["ner/ner.json"])
def example_irex(request, DATADIR):
    with (DATADIR / request.param).open() as f:
        d = json.load(f)
    return d


@pytest.fixture(scope="module", params=["ner/ner2.json"])
def example_ene(request, DATADIR):
    with (DATADIR / request.param).open() as f:
        d = json.load(f)
    return d


@pytest.fixture(scope="module", params=["ner/ner_ene.json"])
def example_ene2(request, DATADIR):
    with (DATADIR / request.param).open() as f:
        d = json.load(f)
    return d


@pytest.fixture(scope="module", params=["ner/long.json"])
def example_long(request, DATADIR):
    with (DATADIR / request.param).open() as f:
        d = json.load(f)
    return d


@pytest.mark.skipif(in_ci(), reason="Fail in circleci due to memory allocation error")
def test_example_batch_irex(nlp_irex: Language, example_irex):
    texts, golds = zip(*example_irex)
    optim = nlp_irex.resume_training()
    nlp_irex.update(texts, golds, optim)


@pytest.mark.skipif(in_ci(), reason="Fail in circleci due to memory allocation error")
def test_example_batch_ene(nlp: Language, example_ene):
    texts, golds = zip(*example_ene)
    optim = nlp.resume_training()
    nlp.update(texts, golds, optim)


def test_long_input(nlp: Language, example_long):
    texts, golds = zip(*example_long)
    optim = nlp.resume_training()
    with pytest.raises(ValueError):
        nlp.update(texts, golds, optim)


@pytest.mark.skipif(in_ci(), reason="Fail in circleci due to memory allocation error")
def test_example_batch_ene2(nlp: Language, example_ene2):
    texts, golds = zip(*example_ene2)
    optim = nlp.resume_training()
    nlp.update(texts, golds, optim)


@pytest.mark.skipif(in_ci(), reason="Fail in circleci due to memory allocation error")
def test_example_batch_ene2_eval(nlp: Language, example_ene2):
    nlp.evaluate(example_ene2)
