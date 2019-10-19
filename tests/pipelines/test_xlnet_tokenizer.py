import pytest
from bedoner.pipelines.wordpiecer import WordPiecer
from pathlib import Path
from spacy.vocab import Vocab
from spacy.tests.util import get_doc
from spacy_transformers.util import ATTRS
from bedoner.lang.trf_mixin import TransformersLanguageMixin


@pytest.fixture(scope="module")
def spiece_model():
    return str(Path(__file__).parent / "../fixtures/xlnet")


@pytest.fixture(scope="module")
def wp_xlnet(spiece_model):
    TransformersLanguageMixin.install_extensions()
    wp = WordPiecer.from_pretrained(Vocab(), spiece_model)
    wp.model.keep_accents = True
    return wp


@pytest.mark.parametrize(
    "tokens,pieces",
    [
        (["今日はいい天気だった"], ["▁", "今日は", "いい", "天気", "だった", "</s>", "<cls>"]),
        (
            ["今日", "は", "いい", "天気", "だっ", "た"],
            [
                "▁今日",
                "▁",
                "は",
                "▁",
                "いい",
                "▁",
                "天気",
                "▁",
                "だ",
                "っ",
                "▁",
                "た",
                "</s>",
                "<cls>",
            ],
        ),
    ],
)
def test_tokenize(wp_xlnet, tokens, pieces):
    doc = get_doc(wp_xlnet.vocab, tokens)
    doc = wp_xlnet(doc)
    assert pieces == doc._.get(ATTRS.word_pieces_)
