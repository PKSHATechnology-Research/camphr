from bedoner.models import bert_ner
from bedoner.entity_extractors.bert_modeling import BertModel
import spacy
import shutil
import pytest
from bedoner.entity_extractors.bert_ner import BertEntityExtractor, create_estimator
from pathlib import Path
import pickle
from spacy.tokens import Doc
import json
from spacy.language import Language
from numpy.testing import assert_array_almost_equal


@pytest.fixture(scope="module")
def nlp():
    return bert_ner()


TESTCASE = [
    (
        "EXILEのATSUSHIと中島美嘉が14日ニューヨーク入り",
        [
            ("ＥＸＩＬＥ", "Show_Organization"),
            ("ＡＴＳＵＳＨＩ", "Person"),
            ("中島美嘉", "Person"),
            ("１４日", "Date"),
            ("ニューヨーク", "City"),
        ],
    ),
    (
        "夏休み真っただ中の8月26日の夕方。大勢の観光客でにぎわうハワイ・ホノルルにあるアラモアナセンターにいたのは藤原竜也（37）だ。",
        [
            ("夏休み", "Date"),
            ("８月２６日", "Date"),
            ("夕方", "Time"),
            ("ハワイ", "Province"),
            ("ホノルル", "City"),
            ("アラモアナセンター", "GOE_Other"),
            ("藤原竜也", "Person"),
            ("３７", "Age"),
        ],
    ),
]


@pytest.mark.parametrize("text,ents", TESTCASE)
def test_call(nlp: Language, text, ents):
    doc: Doc = nlp(text)
    for pred, ans in zip(doc.ents, ents):
        assert pred.text == ans[0]
        assert pred.label_ == ans[1]


def test_pipe(nlp: Language):
    docs = nlp.pipe([text for text, _ in TESTCASE])
    entsl = [ents for _, ents in TESTCASE]
    for doc, ents in zip(docs, entsl):
        for pred, ans in zip(doc.ents, ents):
            assert pred.text == ans[0]
            assert pred.label_ == ans[1]


@pytest.fixture
def saved_nlp(nlp):
    nlp.to_disk("./foo")
    nlp = spacy.load("./foo")
    return nlp


@pytest.mark.parametrize("text,ents", TESTCASE)
def test_to_disk(saved_nlp: Language, text, ents):
    doc: Doc = saved_nlp(text)
    for pred, ans in zip(doc.ents, ents):
        assert pred.text == ans[0]
        assert pred.label_ == ans[1]

