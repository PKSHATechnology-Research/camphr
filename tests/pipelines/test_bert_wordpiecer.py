import spacy
import pytest
from bedoner.models import bert_wordpiecer


@pytest.fixture(scope="module")
def nlp(bert_dir):
    return bert_wordpiecer(lang="juman", pretrained=bert_dir)


@pytest.mark.parametrize(
    "text,pieces",
    [
        (
            "EXILEのATSUSHIと中島美嘉が14日ニューヨーク入り",
            [
                "[CLS]",
                "ＥＸＩＬＥ",
                "の",
                "ＡＴＳ",
                "##ＵＳ",
                "##ＨＩ",
                "と",
                "中島",
                "美",
                "##嘉",
                "が",
                "１４",
                "日",
                "ニューヨーク",
                "入り",
                "[SEP]",
            ],
        )
    ],
)
def test_bert_wordpiecer(nlp, text, pieces):
    doc = nlp(text)
    assert doc._.trf_word_pieces_ == pieces


@pytest.mark.parametrize("lang", ["juman", "mecab"])
def test_bert_wordpiecer_lang(lang, tmpdir, bert_dir):
    nlp = bert_wordpiecer(lang, pretrained=bert_dir)
    assert nlp.meta["lang"] == lang
    d = str(tmpdir.mkdir(lang))
    nlp.to_disk(d)

    nlp2 = spacy.load(d)
    assert nlp2.meta["lang"] == lang
