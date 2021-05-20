import tempfile

from spacy.tests.util import assert_docs_equal
from spacy.vocab import Vocab


def check_juman() -> bool:
    try:
        import pyknp  # noqa
    except ImportError:
        return False
    return True


def check_knp() -> bool:
    return check_juman()


def check_mecab() -> bool:
    try:
        import MeCab  # noqa
    except ImportError:
        return False
    return True


checks = {
    "ja_mecab": check_mecab,
    "ja_juman": check_juman,
    "camphr_torch": lambda: True,
}


def check_lang(lang: str):
    fn = checks.get(lang)
    return fn and fn()


def check_serialization(nlp, text: str = "It is a serialization set. 今日はとてもいい天気だった！"):
    import spacy

    with tempfile.TemporaryDirectory() as d:
        nlp.to_disk(str(d))
        nlp2 = spacy.load(str(d))

        doc1 = nlp(text)
        doc2 = nlp2(text)
        assert_docs_equal(doc1, doc2)
