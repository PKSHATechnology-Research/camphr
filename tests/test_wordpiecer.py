import pytest
from bedoner.lang.juman import Japanese
from pathlib import Path
from spacy.strings import StringStore
from spacy.vocab import Vocab
from spacy_pytorch_transformers.pipeline.wordpiecer import PyTT_WordPiecer
from spacy_pytorch_transformers._tokenizers import SerializableBertTokenizer


@pytest.fixture
def nlp():
    with (
        Path(__file__).parent / "../data/Japanese_L-12_H-768_A-12_E-30_BPE/vocab.txt"
    ).open() as f:
        vs = []
        for line in f:
            vs.append(line[:-1])
    s = StringStore(vs)
    v = Vocab(strings=s)
    nlp = Japanese(v)
    w = PyTT_WordPiecer(v)
    wp = SerializableBertTokenizer(
        str(
            Path(__file__).parent
            / "../data/Japanese_L-12_H-768_A-12_E-30_BPE/vocab.txt"
        ),
        do_lower_case=False,
        tokenize_chinese_chars=False,
    )
    w.model = wp
    nlp.add_pipe(w)
    return nlp


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
    assert doc._.pytt_word_pieces_ == pieces
