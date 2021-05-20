import os
from pathlib import Path

import pytest
from spacy.cli import package
from spacy.language import Language

from camphr_torch.lang import TorchLanguage


@pytest.fixture(scope="session")
def nlp():
    return TorchLanguage(meta={"lang": "en"})


@pytest.fixture
def chdir(tmp_path: Path):
    tmp_path.mkdir(exist_ok=True)
    cwd = os.getcwd()
    os.chdir(tmp_path)
    yield
    os.chdir(cwd)


def test_package(nlp: Language, chdir):
    d = Path().cwd()
    modeld = d / "model"
    pkgd = d / "package"
    pkgd.mkdir()
    nlp.to_disk(modeld)
    package(modeld, pkgd)
