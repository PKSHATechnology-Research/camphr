import json
import tempfile

import pytest
import spacy
import torch
from spacy.gold import GoldParse
from spacy.language import Language
from spacy.tests.util import assert_docs_equal

from bedoner.models import trf_ner
from bedoner.ner_labels.labels_ene import ALL_LABELS as enes
from bedoner.ner_labels.labels_irex import ALL_LABELS as irexes
from bedoner.ner_labels.utils import make_biluo_labels

label_types = ["ene", "irex"]


@pytest.fixture(scope="module", params=label_types)
def label_type(request):
    return request.param


@pytest.fixture(scope="module")
def labels(label_type):
    if label_type == "ene":
        shortenes = [label.split("/")[-1] for label in enes]
        return make_biluo_labels(shortenes)
    elif label_type == "irex":
        return make_biluo_labels(irexes)
    else:
        raise ValueError


@pytest.fixture(scope="module", params=["mecab", "juman", "sentencepiece"])
def nlp(labels, request, trf_dir):
    lang = request.param
    _nlp = trf_ner(lang=lang, labels=labels, pretrained=trf_dir)
    assert _nlp.meta["lang"] == "torch_" + lang
    return _nlp


TESTCASE_ENE = [
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


@pytest.fixture(scope="module", params=["mecab", "juman", "sentencepiece"])
def nlp_for_hooks_test(request, trf_dir):
    lang = request.param
    labels = make_biluo_labels([chr(i) for i in range(65, 91)])

    def convert_label(label: str) -> str:
        if len(label) == 1:
            return label
        return label[:3]

    hook = {"convert_label": convert_label}

    _nlp = trf_ner(lang=lang, labels=labels, pretrained=trf_dir, user_hooks=hook)
    assert lang in _nlp.meta["lang"]
    return _nlp


@pytest.mark.parametrize("text,gold", TESTCASE_ENE)
def test_user_hooks(nlp_for_hooks_test: Language, text, gold, trf_name):
    nlp = nlp_for_hooks_test
    optim = nlp.resume_training()
    nlp.update([text], [gold], optim)


@pytest.mark.parametrize("text,gold", TESTCASE_ENE)
def test_call(nlp: Language, text, gold, label_type):
    if label_type == "irex":
        pytest.skip()
    nlp(text)


def test_serialization(nlp: Language, tmpdir, label_type):
    text = TESTCASE_ENE[0][0]
    if label_type == "irex":
        pytest.skip()
    nlp.to_disk(str(tmpdir))
    nlp2 = spacy.load(str(tmpdir))
    assert_docs_equal(nlp(text), nlp2(text))


def test_pipe(nlp: Language):
    list(nlp.pipe(["今日はいい天気なので外で遊びたい", "明日は晴れ"]))


@pytest.mark.parametrize("text,gold", TESTCASE_ENE)
def test_update(nlp: Language, text, gold, label_type):
    if label_type == "irex":
        pytest.skip()
    assert nlp.device.type == "cpu"
    doc = nlp(text)
    gold = GoldParse(doc, **gold)

    optim = nlp.resume_training()
    assert nlp.device.type == "cpu"
    doc = nlp.make_doc(text)
    assert not doc._.loss
    nlp.update([doc], [gold], optim)
    assert doc._.loss


def test_update_batch(nlp: Language, label_type):
    if label_type == "irex":
        pytest.skip()
    texts, golds = zip(*TESTCASE_ENE)
    optim = nlp.resume_training()
    nlp.update(texts, golds, optim)


def test_evaluate(nlp: Language, label_type):
    if label_type == "irex":
        pytest.skip()
    nlp.evaluate(TESTCASE_ENE)


def test_save_and_load(nlp: Language, label_type):
    if label_type == "irex":
        pytest.skip()
    with tempfile.TemporaryDirectory() as d:
        nlp.to_disk(d)
        nlp = spacy.load(d)
        nlp(TESTCASE_ENE[0][0])


TESTCASE2 = ["資生堂の香水-禅とオードパルファンＺＥＮの違いを教えて下さい。また今でも製造されてますか？"]


@pytest.fixture
def cuda():
    return torch.device("cuda")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda test")
@pytest.mark.parametrize("text,gold", TESTCASE_ENE)
def test_call_cuda(nlp: Language, text, gold, cuda, label_type):
    if label_type == "irex":
        pytest.skip()
    nlp.to(cuda)
    nlp(text)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda test")
@pytest.mark.parametrize("text,gold", TESTCASE_ENE)
def test_update_cuda(nlp: Language, text, gold, cuda, label_type):
    if label_type == "irex":
        pytest.skip()
    nlp.to(cuda)
    doc = nlp(text)
    gold = GoldParse(doc, **gold)

    optim = nlp.resume_training()
    doc = nlp.make_doc(text)
    assert not doc._.loss
    nlp.update([doc], [gold], optim)
    assert doc._.loss


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda test")
def test_update_batch_cuda(nlp: Language, cuda, label_type):
    if label_type == "irex":
        pytest.skip()
    nlp.to(cuda)
    texts, golds = zip(*TESTCASE_ENE)
    optim = nlp.resume_training()
    nlp.update(texts, golds, optim)


@pytest.fixture(
    scope="module",
    params=["ner/ner-ene.json", "ner/ner-irex.json", "ner/ner-ene2.json"],
)
def example_gold(request, DATADIR, label_type):
    fname = request.param
    if label_type in fname:
        with (DATADIR / fname).open() as f:
            d = json.load(f)
        return d
    else:
        pytest.skip()


@pytest.fixture(scope="module", params=["ner/ner-irex-long.json"])
def example_long(request, DATADIR, label_type, trf_name):
    fname = request.param
    if trf_name == "bert" and label_type in fname:
        with (DATADIR / fname).open() as f:
            d = json.load(f)
        return d
    else:
        pytest.skip()


def test_example_batch(nlp: Language, example_gold):
    texts, golds = zip(*example_gold)
    optim = nlp.resume_training()
    nlp.update(texts, golds, optim)


def test_example_batch_eval(nlp: Language, example_gold):
    nlp.evaluate(example_gold)
