from spacy.tokens import Doc
from spacy.vocab import Vocab
from spacy_transformers.util import ATTRS
from bedoner.lang.sentencepiece import SentencePieceLang, EXTS
from bedoner.pipelines.wordpiecer import TrfSentencePiecer
import pytest
from bedoner.models import bert_wordpiecer


@pytest.fixture(scope="module")
def nlp(bert_dir):
    return bert_wordpiecer(lang="juman", pretrained=bert_dir)


@pytest.mark.parametrize(
    "tokens,pieces",
    [(["今日は", "いい", "天気", "だ"], ["▁", "今日は", "いい", "天気", "だ", "</s>", "<cls>"])],
)
def test_wordpiecer(xlnet_wp, tokens, pieces):
    doc = Doc(Vocab(), tokens, spaces=[False] * len(tokens))
    doc = xlnet_wp(doc)
    assert doc._.get(ATTRS.word_pieces_) == pieces


@pytest.fixture
def spm_nlp(spiece_path, xlnet_dir):
    v = Vocab()
    nlp = SentencePieceLang(v, meta={"tokenizer": {"model_path": spiece_path}})
    pipe = TrfSentencePiecer.from_pretrained(v, xlnet_dir)
    nlp.add_pipe(pipe)
    return nlp


@pytest.mark.parametrize("text", ["    New York  ", "今日はいい天気だった", " 今日は\tいい天気　だった"])
def test_trf_sentencepiece(spm_nlp, text):
    nlp = spm_nlp
    doc = nlp(text)
    assert doc._.get(ATTRS.word_pieces)
    assert doc._.get(ATTRS.word_pieces) != doc._.get(EXTS.pieces)
    assert doc._.get(ATTRS.alignment) == doc._.get(EXTS.alignment)
