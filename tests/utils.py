import os
import json
import shutil
from itertools import zip_longest
from typing import Any, Tuple


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
