from camphr.serde import SerDe
import json
import os
import tempfile
from itertools import zip_longest
from pathlib import Path
from typing import Any, Tuple


class Dummy:
    """Dummy class for testing"""

    ...


class DummySerde:
    """Dummy class for testing"""

    @classmethod
    def from_disk(cls, path: Path) -> "DummySerde":
        return DummySerde()

    def to_disk(self, path: Path):
        ...


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


def check_spm() -> bool:
    try:
        import sentencepiece
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


def comp_jsonl(fname1: str, fname2: str) -> Tuple[bool, Any]:
    with open(fname1) as f1, open(fname2) as f2:
        for line1, line2 in zip_longest(f1, f2, fillvalue=[]):
            d1 = json.loads(line1)
            d2 = json.loads(line2)
            if d1 != d2:
                return False, (d1, d2)
    return True, ()


def in_ci():
    return os.getenv("CI", "") == "true"


def check_serialization(nlp, text: str = "It is a serialization set. 今日はとてもいい天気だった！"):
    with tempfile.TemporaryDirectory() as d:
        nlp.to_disk(str(d))
        nlp2 = spacy.load(str(d))
        assert_docs_equal(nlp(text), nlp2(text))


FIXTURE_DIR = (Path(__file__).parent / "fixtures/").absolute()
BERT_JA_DIR = FIXTURE_DIR / "bert-base-japanese-test"
BERT_DIR = FIXTURE_DIR / "bert-test"
XLNET_DIR = FIXTURE_DIR / "xlnet"
DATA_DIR = (Path(__file__).parent / "data/").absolute()

TRF_TESTMODEL_PATH = [str(BERT_JA_DIR), str(XLNET_DIR), str(BERT_DIR)]
LARGE_MODELS = {"albert-base-v2"}
