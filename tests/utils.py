import json
import os
import shutil
import tempfile
from itertools import zip_longest
from pathlib import Path
from typing import Any, Tuple

import spacy
from spacy.tests.util import assert_docs_equal


def check_juman():
    return shutil.which("juman") is not None


def check_knp():
    return shutil.which("knp") is not None


def check_mecab():
    return shutil.which("mecab") is not None


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


def check_serialization(nlp, text: str = "今日は，とてもいい天気だった!"):
    with tempfile.TemporaryDirectory() as d:
        nlp.to_disk(str(d))
        nlp2 = spacy.load(str(d))
        assert_docs_equal(nlp(text), nlp2(text))


FIXTURE_DIR = (Path(__file__).parent / "fixtures/").absolute()
BERT_DIR = FIXTURE_DIR / "bert-base-japanese-test"
XLNET_DIR = FIXTURE_DIR / "xlnet"
DATA_DIR = (Path(__file__).parent / "data/").absolute()
