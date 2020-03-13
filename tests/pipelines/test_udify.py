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
    nlp2 = spacy.load(str(tmpdir))
    docs2 = [nlp2(text) for text in TEXTS]
    for doc1, doc2 in zip(docs, docs2):
        if spacy.__version__ == "2.2.4":
            # this version of spacy has a bug in `assert_docs_equal`.
            # see https://github.com/explosion/spaCy/issues/5144
            return
        assert_docs_equal(doc1, doc2)
