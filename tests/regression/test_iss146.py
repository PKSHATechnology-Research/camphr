import pytest
from camphr.lang.sentencepiece import TorchSentencePieceLang
from camphr.pipelines.trf_model import BertModel
from camphr.pipelines.wordpiecer import TrfSentencePiecer
from spacy_transformers._tokenizers import SerializableBertTokenizer


@pytest.fixture(scope="module")
def nlp(spiece_path, fixture_dir, bert_dir):
    _nlp = TorchSentencePieceLang(meta={"tokenizer": {"model_path": str(spiece_path)}})

    t = SerializableBertTokenizer(str(fixture_dir / "spiece_words.txt"))
    pipes = []
    pipe = TrfSentencePiecer(_nlp.vocab, t, trf_name="bert")
    pipes.append(pipe)

    pipe = BertModel.from_pretrained(_nlp.vocab, bert_dir)
    pipes.append(pipe)

    for pipe in pipes:
        _nlp.add_pipe(pipe)
    return _nlp


@pytest.mark.parametrize("text", ["K"])
def test_forward(nlp, text):
    nlp(text)
