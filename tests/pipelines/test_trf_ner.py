import json

import pytest
from camphr.models import trf_ner
from camphr.ner_labels.labels_ene import ALL_LABELS as enes
from camphr.ner_labels.labels_irex import ALL_LABELS as irexes
from camphr.ner_labels.utils import make_biluo_labels
from spacy.language import Language

from ..utils import DATA_DIR, check_serialization

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


@pytest.fixture(scope="module")
def nlp(labels, torch_lang, trf_name_or_path, device):
    _nlp = trf_ner(lang=torch_lang, labels=labels, pretrained=trf_name_or_path)
    assert _nlp.meta["lang"] == torch_lang
    _nlp.to(device)
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


@pytest.mark.parametrize("text,gold", TESTCASE_ENE)
def test_call(nlp: Language, text, gold, label_type):
    if label_type == "irex":
        pytest.skip()
    nlp(text)


def test_update(nlp: Language, label_type):
    if label_type == "irex":
        pytest.skip()
    optim = nlp.resume_training()
    nlp.update(*zip(*TESTCASE_ENE), optim)


@pytest.fixture(scope="module", params=["ja_mecab_torch", "ja_juman_torch"])
def nlp_for_hooks_test(request, trf_name_or_path):
    lang = request.param
    labels = make_biluo_labels([chr(i) for i in range(65, 91)])

    def convert_label(label: str) -> str:
        if len(label) == 1:
            return label
        return label[:3]

    hook = {"convert_label": convert_label}

    _nlp = trf_ner(
        lang=lang, labels=labels, pretrained=trf_name_or_path, user_hooks=hook
    )
    assert lang in _nlp.meta["lang"]
    return _nlp


@pytest.mark.parametrize("text,gold", TESTCASE_ENE)
def test_user_hooks(nlp_for_hooks_test: Language, text, gold):
    nlp = nlp_for_hooks_test
    optim = nlp.resume_training()
    nlp.update([text], [gold], optim)


@pytest.fixture(
    scope="module",
    params=["ner/ner-ene.json", "ner/ner-irex.json", "ner/ner-ene2.json"],
)
def example_gold(request, label_type):
    fname = request.param
    if label_type in fname:
        with (DATA_DIR / fname).open() as f:
            d = json.load(f)
        return d
    else:
        pytest.skip()


@pytest.fixture(scope="module", params=["ner/ner-irex-long.json"])
def example_long(request, label_type, trf_name_or_path):
    fname = request.param
    if label_type in fname:
        with (DATA_DIR / fname).open() as f:
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


def test_freeze_ner(trf_name_or_path):
    nlp = trf_ner("ja_mecab_torch", trf_name_or_path, freeze=True, labels=["foo"])
    pipe = nlp.pipeline[-2][1]
    assert pipe.cfg["freeze"]


def test_serialization(nlp):
    check_serialization(nlp)
