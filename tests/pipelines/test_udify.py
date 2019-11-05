from pathlib import Path
import pytest
import spacy
from spacy.tests.util import assert_docs_equal
from bedoner.pipelines.udify import Udify
import bedoner.lang.mecab as mecab
from ..utils import in_ci

pytestmark = pytest.mark.skipif(in_ci(), reason="heavy test")


@pytest.fixture(scope="module")
def nlp():
    _nlp = mecab.Japanese()
    pipe = Udify.from_archive(Path(__file__).parent / "../../data/udify")
    _nlp.add_pipe(pipe)
    return _nlp


TEXTS = ["駅から遠く、お酒を楽しむには不便なリッチなのが最大のネックですが、ドライバーを一人連れてでも行きたいお店です☆", "今日はいい天気だった"]


@pytest.mark.parametrize("text", TEXTS)
def test_udify(nlp, text):
    doc = nlp(text)
    assert doc.is_parsed


def test_serialization(nlp, tmpdir):
    docs = [nlp(text) for text in TEXTS]
    for i in range(2):
        d = str(tmpdir + f"/{i}")
        nlp.to_disk(d)
        nlp = spacy.load(d)
        docs2 = [nlp(text) for text in TEXTS]
        for doc1, doc2 in zip(docs, docs2):
            assert_docs_equal(doc1, doc2)
