from spacy.tokens import Doc
from spacy.vocab import Vocab
from spacy_transformers.util import ATTRS
import pytest


@pytest.mark.parametrize(
    "tokens,pieces",
    [(["今日は", "いい", "天気", "だ"], ["▁", "今日は", "いい", "天気", "だ", "</s>", "<cls>"])],
)
def test_wordpiecer(xlnet_wp, tokens, pieces):
    doc = Doc(Vocab(), tokens, spaces=[False] * len(tokens))
    doc = xlnet_wp(doc)
    assert doc._.get(ATTRS.word_pieces_) == pieces
