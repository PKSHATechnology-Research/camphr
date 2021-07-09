import time
import subprocess
import sys
from typing import Dict


PYTHON_VERSIONS = ["3.7", "3.8", "3.9"]
PACKAGES = ["None", "camphr_transformers"]
EXTRAS = {
    "None": [
        ("None", ""),
        ("mecab", "-E mecab"),
        ("juman", "-E juman"),
        ("sentencepiece", "-E sentencepiece"),
    ],
    "camphr_transformers": [("None", "-E torch")],
}


procs: Dict[str, subprocess.Popen] = {}
stats: Dict[str, bool] = {}
for version in PYTHON_VERSIONS:
    for package in PACKAGES:
        for dockerfile_ext, poetry_arg in EXTRAS[package]:
            install_cmd = f"poetry install {poetry_arg}"
            cmd = [
                "python",
                "test.py",
                version,
                "--package",
                package,
                "--dockerfile_ext",
                dockerfile_ext,
                "--install_cmd",
                install_cmd,
            ] + sys.argv[1:]
            p = subprocess.Popen(cmd)
            key = " ".join(cmd)
            procs[key] = p
            stats[key] = False


def terminate():
    for proc in procs.values():
        proc.kill()


while True:
    for key, proc in procs.items():
        if stats[key]:
            continue
        stat = proc.poll()
        if stat is not None:
            if stat == 0:
                stats[key] = True
            else:
                terminate()
                raise ValueError(f"Failed: {key}")
    time.sleep(1)
