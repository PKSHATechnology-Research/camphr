import pytest


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
def test_bert_wordpiecer(bert_wordpiece_nlp, text, pieces):
    doc = bert_wordpiece_nlp(text)
    assert doc._.pytt_word_pieces_ == pieces
