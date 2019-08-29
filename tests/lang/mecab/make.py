"""テストケース作るためのやつ"""
from test_tokenizer import TOKENIZER_TESTS
import subprocess

for t in TOKENIZER_TESTS:
    out = subprocess.check_output(f"echo {t[0]} | mecab", shell=True).decode()
    surfl, posl = [], []
    for line in out.split("\n"):
        fs = line.split("\t")
        if len(fs) == 1:
            continue
        surf, rest = fs
        surfl.append(surf)
        posl.append(",".join(rest.split(",")[0:4]))
    print((t[0], posl), ",")

