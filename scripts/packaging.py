import tempfile
import json
import spacy
import subprocess
from pathlib import Path
from spacy.language import Language
from typing import List, Any, Dict, TypeVar, Union
import omegaconf

from spacy.cli import package
import hydra
import logging

log = logging.Logger(__name__)


__dir__ = Path(__file__).parent
PACKAGES_DIR = __dir__ / "../pkgs"
LANG_REQUIREMTNS = {"juman": ["pyknp"], "mecab": ["mecab-python3"], "knp": ["pyknp"]}
THIS_MODULE = "bedoner @ git+https://github.com/PKSHATechnology/bedore-ner@{ref}"
Pathlike = Union[str, Path]


def get_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode()


def requirements(meta: Dict[str, Any], ref_module: str = "") -> List[str]:
    lang = meta["lang"]
    req = meta.get("requirements") or []
    req += LANG_REQUIREMTNS.get(lang, [])
    req.append(THIS_MODULE.format(ref=ref_module))
    return req


def create_package_from_nlp(nlp: Language, pkgsdir: Pathlike) -> Path:
    pkgsdir = Path(pkgsdir)
    tmpd = tempfile.TemporaryDirectory()
    nlp.to_disk(str(tmpd.name))
    return create_package(tmpd.name, pkgsdir, nlp.meta)


def meta_to_modelname(meta: Dict[str, Any]) -> str:
    return meta["lang"] + "_" + meta["name"]


def create_package(
    model_dir: Pathlike, pkgsdir: Pathlike, meta: Dict[str, Any]
) -> Path:
    package(model_dir, pkgsdir, force=True)
    model_name = meta_to_modelname(meta)
    pkgd = pkgsdir / (model_name + "-" + meta["version"])
    return pkgd


class Config(omegaconf.Config):
    model: str  # path to model dir
    ref_module: str = ""  # git ref to bedoner
    packages_dir: Pathlike = ""  # directory containing models
    version: str = ""  # model version. Should be same as git tag.


T = TypeVar("T")


def log_val(obj: T, head: str = "", logtype="info") -> T:
    msg = head + f": {obj}"
    getattr(log, logtype)(msg)
    return obj


def get_meta(model_dir: Pathlike) -> Dict[str, Any]:
    model_dir = Path(model_dir)
    with (model_dir / "meta.json").open() as f:
        meta = json.load(f)
    return meta


def set_meta(model_dir: Pathlike, meta: Dict[str, Any]):
    model_dir = Path(model_dir)
    with (model_dir / "meta.json").open("w") as f:
        json.dump(meta, f)


@hydra.main()
def main(cfg: Config):
    cfg.ref_module = cfg.get("ref_module") or get_commit()
    cfg.packages_dir = cfg.get("packages_dir") or PACKAGES_DIR

    meta = get_meta(cfg.model)
    cfg.version = meta.setdefault("version", cfg.version)
    assert cfg.version, "You should specify model version, e.g. version=1.0.0.dev3"
    log.info(cfg.pretty())

    meta["requirements"] = log_val(requirements(meta), "Requirements")

    pkgsdir = cfg.packages_dir
    if isinstance(pkgsdir, str):
        pkgsdir = Path(pkgsdir)
    pkgsdir.mkdir(exist_ok=True)

    log_val(create_package(cfg.model, pkgsdir, meta), "Saved")


if __name__ == "__main__":
    main()
