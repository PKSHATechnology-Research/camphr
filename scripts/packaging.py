from pathlib import Path
import tempfile
from spacy.cli import package

additional_requirements = [
    "bedoner @ git+https://github.com/PKSHATechnology/bedore-ner"
]

pkgs = Path(__file__).parent / "../pkgs"


def create_package(nlp):
    meta = nlp.meta
    req = meta.get("requirements") or []
    req.append(additional_requirements)
    nlp.meta["requirements"] = req

    tmpd = tempfile.TemporaryDirectory()
    nlp.to_disk(str(tmpd.name))
    package(tmpd.name, pkgs, force=True)
    model_name = meta["lang"] + "_" + meta["name"]
    pkgd = pkgs / (model_name + "-" + meta["version"])
    return pkgd, tmpd
