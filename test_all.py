import subprocess
import sys
from typing import Dict


PYTHON_VERSIONS = ["3.7", "3.8", "3.9"]
PACKAGES = ["None", "camphr_transformers"]
EXTRAS = ["None", "mecab", "juman", "sentencepiece"]

procs: Dict[str, subprocess.Popen] = {}
for version in PYTHON_VERSIONS:
    for package in PACKAGES:
        for extra in EXTRAS:
            cmd = [
                "python",
                "test.py",
                version,
                "--package",
                package,
                "--extra",
                extra,
            ] + sys.argv[1:]
            p = subprocess.Popen(cmd)
            procs[" ".join(cmd)] = p

for k, p in procs.items():
    print(k)
    p.wait()
    if p.returncode != 0:
        # cleanup
        for proc in procs.values():
            proc.kill()
        raise ValueError(f"Faild: {k}")
