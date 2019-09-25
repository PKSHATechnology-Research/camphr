import tempfile

import pytest
import spacy
from bedoner.models import bert_ner
from bedoner.ner_labels.labels_ene import ALL_LABELS
from bedoner.ner_labels.labels_irex import ALL_LABELS as irexs
from bedoner.ner_labels.utils import make_biluo_labels
from spacy.gold import GoldParse, spans_from_biluo_tags
from spacy.language import Language
from spacy.tokens import Doc


@pytest.fixture
def labels():
    return make_biluo_labels(ALL_LABELS)


@pytest.fixture
def nlp(labels):
    return bert_ner(labels=["-"] + labels)


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
]


@pytest.mark.parametrize("text,gold", TESTCASE)
def test_call(nlp: Language, text, gold):
    nlp(text)


def test_pipe(nlp: Language):
    list(nlp.pipe(["今日はいい天気なので外で遊びたい", "明日は晴れ"]))


def is_same_ner(doc: Doc, gold: GoldParse) -> bool:
    if len(doc) != len(gold.ner):
        return False
    gold_spans = spans_from_biluo_tags(doc, gold.ner)
    if len(gold_spans) != len(doc.ents):
        return False
    res = True
    for e, e2 in zip(doc.ents, gold_spans):
        res &= e == e2
    return res


@pytest.mark.parametrize("text,gold", TESTCASE)
def test_update(nlp: Language, text, gold):
    doc = nlp(text)
    gold = GoldParse(doc, **gold)
    assert not is_same_ner(doc, gold)

    optim = nlp.resume_training()
    doc = nlp.make_doc(text)
    assert doc._.loss is None
    nlp.update([doc], [gold], optim)
    assert doc._.loss


def test_update_batch(nlp: Language):
    texts, golds = zip(*TESTCASE)
    optim = nlp.resume_training()
    nlp.update(texts, golds, optim)


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
