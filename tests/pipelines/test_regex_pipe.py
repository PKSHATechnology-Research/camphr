import pytest
from bedoner.pipelines.regex_ner import postcode_ruler

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
