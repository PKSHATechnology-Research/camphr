from pathlib import Path

import bedoner.lang.mecab as mecab
import pytest
from bedoner.pipelines.elmo import Elmo

from ..utils import in_ci

pytestmark = pytest.mark.skipif(in_ci(), reason="heavy test")


@pytest.fixture(scope="module")
def nlp():
    _nlp = mecab.Japanese()
    elmod = Path(__file__).parent / "../../data/elmo"
    options = elmod / "options.json"
    weights = elmod / "weights.hdf5"
    m = Elmo.Model(options, weights)
    p = Elmo(model=m)
    _nlp.add_pipe(p)
    return _nlp


@pytest.mark.parametrize("text", ["私は文脈ベクトルの元祖です"])
def test_elmo(nlp, text):
    doc = nlp(text)
    assert doc.tensor.shape[0] == len(doc)
    assert doc.vector is not None
    assert doc[0].vector is not None
    assert doc[1:].vector is not None
