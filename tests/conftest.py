import pytest
from bedoner.lang.mecab import Japanese as Mecab
from bedoner.lang.juman import Japanese as Juman
from bedoner.lang.knp import Japanese as KNP
from pathlib import Path
from bedoner.lang.juman import Japanese
from pathlib import Path
from spacy.strings import StringStore
from spacy.vocab import Vocab
from spacy_pytorch_transformers.pipeline.wordpiecer import PyTT_WordPiecer
from spacy_pytorch_transformers._tokenizers import SerializableBertTokenizer


@pytest.fixture(scope="session")
def mecab_tokenizer():
    return Mecab.Defaults.create_tokenizer()


@pytest.fixture(scope="session")
def juman_tokenizer():
    return Juman.Defaults.create_tokenizer(juman_kwargs={"jumanpp": False})


@pytest.fixture(scope="session")
def jumanpp_tokenizer():
    return Juman.Defaults.create_tokenizer(juman_kwargs={"jumanpp": True})


@pytest.fixture(scope="session")
def knp_tokenizer():
    return KNP.Defaults.create_tokenizer()


@pytest.fixture(scope="session")
def bert_wordpiece_nlp():
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
