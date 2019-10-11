import spacy
import py
import pytest
from bedoner.models import bert_wordpiecer


@pytest.mark.parametrize("lang", ["juman", "mecab"])
def test_bert_wordpiecer(lang, tmpdir):
    nlp = bert_wordpiecer(lang)
    assert nlp.meta["lang"] == lang
    d = str(tmpdir.mkdir(lang))
    nlp.to_disk(d)

    nlp2 = spacy.load(d)
    assert nlp2.meta["lang"] == lang

