from itertools import zip_longest

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


@pytest.mark.parametrize("text,roots", zip(TEXTS, [["involve"], ["Who", "am"]]))
def test_udify(nlp: Language, text, roots):
    doc = nlp(text)
    assert doc.is_parsed
    for s, root in zip_longest(doc.sents, roots):
        assert s.root.text == root


def test_serialization(nlp, tmpdir):
    docs = [nlp(text) for text in TEXTS]
    for i in range(2):
        d = str(tmpdir + f"/{i}")
        nlp.to_disk(d)
        nlp = spacy.load(d)
        docs2 = [nlp(text) for text in TEXTS]
        for doc1, doc2 in zip(docs, docs2):
            assert_docs_equal(doc1, doc2)
