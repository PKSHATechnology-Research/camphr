from bedoner.lang import mecab
import pytest
from spacy.language import Language
from spacy.tokens import Doc, Span

from bedoner.pipelines.regex_ruler import (
    carcode_ruler,
    RegexRuler,
    postcode_ruler,
    DateRuler,
    LABEL_DATE,
)

from ..utils import check_mecab

pytestmark = pytest.mark.skipif(
    not check_mecab(), reason="mecab is not always necessary"
)

TESTCASES_POSTCODE = [
    ("〒100-0001", ["〒100-0001"]),
    ("100-0001", ["100-0001"]),
    ("1000-0001", []),
    ("100-00001", []),
    ("080-1234-5678", []),
    ("〒100-0001　東京都千代田区千代田1-1", ["〒100-0001"]),
]


@pytest.mark.parametrize("text,expecteds", TESTCASES_POSTCODE)
def test_postcode_ruler(mecab_tokenizer, text, expecteds):
    doc = mecab_tokenizer(text)
    doc = postcode_ruler(doc)
    assert len(doc.ents) == len(expecteds)
    for ent, expected in zip(doc.ents, expecteds):
        assert ent.text == expected


def test_postcode_merge(mecab_tokenizer):
    text = "〒100-0001 234-5678"
    doc = mecab_tokenizer(text)
    postcode_ruler.merge = True
    doc = postcode_ruler(doc)
    assert len(doc.ents) == 2
    assert len(doc) == 2


TESTCASES_CARCODE = [
    ("自動車番号: 品川500 さ 2345", ["品川500 さ 2345"]),
    ("自動車番号: 品川500 さ 2345, 横浜500 し 3456", ["品川500 さ 2345", "横浜500 し 3456"]),
]

TESTCASES_CARCODE_TODO = [("以下の契約の明細003と019について質問させていただきます。", [])]


@pytest.mark.parametrize("text,expecteds", TESTCASES_CARCODE)
def test_carcode_ruler(mecab_tokenizer, text, expecteds):
    doc = mecab_tokenizer(text)
    doc = carcode_ruler(doc)
    assert len(doc.ents) == len(expecteds)
    for ent, expected in zip(doc.ents, expecteds):
        assert ent.text == expected


@pytest.mark.xfail(reason="TODO", strict=True)
@pytest.mark.parametrize("text,expecteds", TESTCASES_CARCODE_TODO)
def test_carcode_ruler_todo(mecab_tokenizer, text, expecteds):
    doc = mecab_tokenizer(text)
    doc = carcode_ruler(doc)
    assert len(doc.ents) == len(expecteds)
    for ent, expected in zip(doc.ents, expecteds):
        assert ent.text == expected


@pytest.fixture
def nlp():
    return mecab.Japanese()


def test_compose_pipes(nlp: Language):
    nlp.add_pipe(carcode_ruler)
    nlp.add_pipe(postcode_ruler)
    text = "郵便番号は〒100-0001で，車の番号は品川500 さ 2345です"
    doc = nlp(text)
    assert len(doc.ents) == 2


def test_conflict_label(nlp: Language):
    text = "郵便番号は〒100-0001で，車の番号は品川500 さ 2345です"
    nlp.add_pipe(carcode_ruler)
    nlp.add_pipe(RegexRuler(r"\d*", label="NUMBER"))
    doc = nlp(text)
    assert len(doc.ents) == 4


@pytest.fixture
def nlp_with_date(nlp):
    nlp.add_pipe(DateRuler())
    return nlp


TESTS = [("今日は2019年11月30日だ", "2019年11月30日"), ("僕は平成元年4月10日生まれだ", "平成元年4月10日")]


@pytest.mark.parametrize("text,ent", TESTS)
def test_date_entity_ruler(nlp_with_date: Language, text: str, ent: str):
    doc: Doc = nlp_with_date(text)
    assert len(doc.ents) == 1
    span: Span = doc.ents[0]
    assert span.text == ent
    assert span.label_ == LABEL_DATE
