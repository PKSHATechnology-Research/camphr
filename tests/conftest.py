import pytest
from bedoner.lang.mecab import Japanese as Mecab
from bedoner.lang.juman import Japanese as Juman
from bedoner.lang.knp import Japanese as KNP
from pathlib import Path
from pathlib import Path
from spacy.strings import StringStore
from spacy.vocab import Vocab
from bedoner.wordpiecer import BertWordPiecer


@pytest.fixture(scope="session")
def mecab_tokenizer():
    return Mecab.Defaults.create_tokenizer(opt="-d /usr/local/lib/mecab/dic/ipadic")


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
    nlp = Juman(v)
    w = BertWordPiecer(
        v,
        vocab_file=str(
            Path(__file__).parent
            / "../data/Japanese_L-12_H-768_A-12_E-30_BPE/vocab.txt"
        ),
    )
    w.model = w.Model(w.cfg["vocab_file"])
    nlp.add_pipe(w)
    return nlp
