from pathlib import Path

import pytest
import spacy

import camphr.lang.mecab as mecab
from camphr.pipelines.elmo import Elmo

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def nlp():
    _nlp = mecab.Japanese()
    elmod = Path(__file__).parent / "../../data/elmo"
    options = elmod / "options.json"
    weights = elmod / "weights.hdf5"
    elmo = Elmo.from_elmofiles(options, weights)
    _nlp.add_pipe(elmo)
    return _nlp


TEXTS = ["私は文脈ベクトルの元祖です"]


@pytest.mark.parametrize("text", TEXTS)
def test_elmo(nlp, text):
    doc = nlp(text)
    assert doc.tensor.shape[0] == len(doc)
    assert doc.vector is not None
    assert doc[0].vector is not None
    assert doc[1:].vector is not None


def test_serialization(nlp, tmpdir):
    text = TEXTS[0]
    doc = nlp(text)
    nlp.to_disk(str(tmpdir))
    nlp = spacy.load(str(tmpdir))
    assert doc.tensor.shape == nlp(text).tensor.shape
