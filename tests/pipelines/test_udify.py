from pathlib import Path
import pytest
from bedoner.pipelines.udify import Udify
import bedoner.lang.mecab as mecab
from ..utils import in_ci

pytestmark = pytest.mark.skipif(in_ci(), reason="heavy test")


@pytest.fixture
def nlp():
    _nlp = mecab.Japanese()
    pipe = Udify.from_archive(Path(__file__).parent / "../../data/udify")
    _nlp.add_pipe(pipe)
    return _nlp


def test_udify(nlp):
    doc = nlp("今日はいい天気だった")
    assert doc.is_parsed
