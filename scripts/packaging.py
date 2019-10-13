import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, TypeVar, Union

import hydra
import omegaconf
from spacy.cli import package
from spacy.language import Language

log = logging.getLogger(__name__)


__dir__ = Path(__file__).parent
PACKAGES_DIR = __dir__ / "../pkgs"
LANG_REQUIREMTNS = {"juman": {"pyknp"}, "mecab": {"mecab-python3"}, "knp": {"pyknp"}}
THIS_MODULE = "bedoner @ git+https://github.com/PKSHATechnology/bedore-ner@{ref}"
Pathlike = Union[str, Path]


def get_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()


def requirements(meta: Dict[str, Any], ref_module: str) -> List[str]:
    lang = meta["lang"]
    req = set(meta.get("requirements", {}))
    req |= LANG_REQUIREMTNS.get(lang, {})
    req = {k for k in req if not k.startswith("bedoner")}
    req.add(THIS_MODULE.format(ref=ref_module))
    return list(req)


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
    pkgsdir = Path(pkgsdir)
    package(model_dir, pkgsdir, force=True)
    model_name = meta_to_modelname(meta)
    pkgd = pkgsdir / (model_name + "-" + meta["version"])
    return pkgd


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


def abs_path(path: str) -> str:
    return os.path.abspath(os.path.join("../../../../", path))


class Config(omegaconf.Config):
    model: str  # path to model dir
    ref_module: str = ""  # git ref to bedoner
    packages_dir: Pathlike = ""  # directory containing models
    version: str = ""  # model version. Should be same as git tag.


@hydra.main(config_path="conf/packaging.yml")
def main(cfg: Config):
    assert cfg.model, "Model path is required, e.g. model=path_to_model_dir"
    cfg.model = abs_path(cfg.model)
    cfg.ref_module = cfg.ref_module or get_commit()
    cfg.packages_dir = abs_path(cfg.packages_dir or str(PACKAGES_DIR))

    meta = get_meta(cfg.model)
    cfg.version = cfg.version or meta.get("version", "")
    assert cfg.version, "Model version is required, e.g. version=1.0.0.dev3"
    meta["version"] = cfg.version
    log.info(cfg.pretty())

    meta["requirements"] = log_val(requirements(meta, cfg.ref_module), "Requirements")
    set_meta(cfg.model, meta)

    Path(cfg.packages_dir).mkdir(exist_ok=True)

    log_val(create_package(cfg.model, cfg.packages_dir, meta), "Saved")


if __name__ == "__main__":
    main()
