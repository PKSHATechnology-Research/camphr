import pytest
import spacy
from spacy.language import Language
from spacy.tests.util import assert_docs_equal

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def nlp():
    return spacy.load("en_udify")


TEXTS = [
    "Challenges in natural language processing frequently involve speech recognition.",
    "Who are you. I am Udify.",
]


@pytest.mark.parametrize("text", TEXTS)
def test_udify(nlp: Language, text):
    doc = nlp(text)
    assert doc.is_parsed
    for s in doc.sents:
        assert s.root.text


def test_serialization(nlp, tmpdir):
    docs = [nlp(text) for text in TEXTS]
    nlp.to_disk(str(tmpdir))
    nlp = spacy.load(str(tmpdir))
    docs2 = [nlp(text) for text in TEXTS]
    for doc1, doc2 in zip(docs, docs2):
        assert_docs_equal(doc1, doc2)
