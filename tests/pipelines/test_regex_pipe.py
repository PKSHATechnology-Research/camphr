import pytest

from bedoner.pipelines.regex_ruler import carcode_ruler, postcode_ruler

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
