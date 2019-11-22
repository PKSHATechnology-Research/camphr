"""

Usage:

$ python versionup_package.py --model model_path --version new_version
"""
import os
import subprocess
import sys

import fire
import spacy
from packaging_nlp import PACKAGES_DIR, create_package_from_nlp, requirements
from spacy.language import Language


def create_nlp(model: str, version: str) -> Language:
    nlp: Language = spacy.load(model)
    nlp.meta["version"] = version
    nlp.meta["requirements"] = requirements(nlp.meta, version)
    return nlp


def main(model: str, version: str, sdist: bool = True):
    nlp = create_nlp(model, version)
    print("Loaded model")
    pkgd = create_package_from_nlp(nlp, PACKAGES_DIR)
    print("Created package")
    if sdist:
        os.chdir(pkgd)
        print(subprocess.check_output([sys.executable, "setup.py", "sdist"]).decode())
        print(f"Created sdist in {(pkgd / 'dist').absolute()}")


if __name__ == "__main__":
    fire.Fire(main)
