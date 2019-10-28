import subprocess
from pathlib import Path
import fire
from itertools import product


basedir = Path(__file__).parent


def main(lang="mecab", pretrained=""):
    if not pretrained:
        pretrained = str((basedir / "../tests/fixtures/bert").absolute())
    cfg = {
        "data": str((basedir / "../tests/data/ner/gold-ene.jsonl").absolute()),
        "pretrained": pretrained,
        "lang": lang,
        "niter": 2,
        "neval": 1,
    }
    cmd = ["python", str(basedir / "train.py")]
    for k, v in cfg.items():
        cmd.append(f"{k}={v}")
    p = subprocess.run(cmd)
    assert p.returncode == 0, (lang, pretrained)


def matrix():
    fixturesdir = basedir / "../tests/fixtures"
    langs = ["mecab", "juman", "sentencepiece"]
    pretrained_dirs = [
        str((fixturesdir / "bert").absolute()),
        str((fixturesdir / "xlnet").absolute()),
    ]
    for lang, pretrained in product(langs, pretrained_dirs):
        main(lang, pretrained)


if __name__ == "__main__":
    fire.Fire()
