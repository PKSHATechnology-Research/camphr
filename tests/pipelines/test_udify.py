from itertools import zip_longest
from pathlib import Path

import bedoner.lang.mecab as mecab
import pytest
import spacy
from bedoner.pipelines.udify import Udify
from spacy.tests.util import assert_docs_equal
from spacy.tokens import Doc

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def nlp():
    _nlp = mecab.Japanese()
    sentencizer = _nlp.create_pipe("sentencizer")
    sentencizer.punct_chars.add("。")
    _nlp.add_pipe(sentencizer)

    pipe = Udify.from_archive(Path(__file__).parent / "../../data/udify")
    _nlp.add_pipe(pipe)
    return _nlp


TEXTS = [
    "駅から遠く、お酒を楽しむには不便なリッチなのが最大のネックですが、ドライバーを一人連れてでも行きたいお店です",
    "今日はいい天気だった。明日は晴れるかな",
]


@pytest.mark.parametrize("text,roots", zip(TEXTS, [["店"], ["天気", "晴れる"]]))
def test_udify(nlp, text, roots):
    doc: Doc = nlp(text)
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
