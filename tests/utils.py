from itertools import zip_longest
import json
import os
from pathlib import Path
from typing import Any, Tuple


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


FIXTURE_DIR = (Path(__file__).parent / "fixtures/").absolute()
BERT_DIR = FIXTURE_DIR / "bert-test"
BERT_JA_DIR = FIXTURE_DIR / "bert-base-japanese-test"
XLNET_DIR = FIXTURE_DIR / "xlnet"
DATA_DIR = (Path(__file__).parent / "data/").absolute()

TRF_TESTMODEL_PATH = [str(BERT_JA_DIR), str(XLNET_DIR), str(BERT_DIR)]
LARGE_MODELS = {"albert-base-v2"}
