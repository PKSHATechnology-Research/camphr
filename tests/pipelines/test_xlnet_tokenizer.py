import pytest
from spacy.tokens import Doc
from spacy_transformers.util import ATTRS


@pytest.mark.parametrize(
    "tokens,pieces",
    [(["今日は", "いい", "天気", "だった"], ["▁", "今日は", "いい", "天気", "だった", "</s>", "<cls>"])],
)
def test_tokenize(xlnet_wp, tokens, pieces):
    doc = Doc(xlnet_wp.vocab, tokens, spaces=[False] * len(tokens))
    doc = xlnet_wp(doc)
    assert pieces == doc._.get(ATTRS.word_pieces_)
