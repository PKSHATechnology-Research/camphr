import json

import omegaconf
import pytest
from camphr.models import create_model
from camphr.ner_labels.labels_ene import ALL_LABELS as enes
from camphr.ner_labels.labels_irex import ALL_LABELS as irexes
from camphr.ner_labels.utils import make_biluo_labels
from camphr.pipelines.trf_model import TRANSFORMERS_MODEL
from camphr.pipelines.trf_ner import TRANSFORMERS_NER
from camphr.pipelines.trf_tokenizer import TRANSFORMERS_TOKENIZER
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
def config(labels, lang, trf_name_or_path, device):
    return omegaconf.OmegaConf.create(
        f"""
    lang:
        name: {lang}
        torch: true
        optimizer:
            class: torch.optim.SGD
            lr: 0.01
    pipeline:
        {TRANSFORMERS_TOKENIZER}:
            trf_name_or_path: {trf_name_or_path}
        {TRANSFORMERS_MODEL}:
            trf_name_or_path: {trf_name_or_path}
        {TRANSFORMERS_NER}:
            trf_name_or_path: {trf_name_or_path}
            labels: {labels}
    """
    )


@pytest.fixture(scope="module")
def nlp(config, device):
    _nlp = create_model(config)
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


def test_serialization(nlp):
    check_serialization(nlp)
