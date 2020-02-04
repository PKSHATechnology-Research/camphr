import pytest
import spacy

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def nlp():
    return spacy.load("en_elmo_medium")


TEXTS = ["私は文脈ベクトルの元祖です", "ELMo is a deep contextualized word representation"]


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
