from pathlib import Path

import pytest
from spacy.cli import package
from spacy.language import Language

from camphr_core.lang.torch import TorchLanguage


@pytest.fixture(scope="session")
def nlp():
    return TorchLanguage(meta={"lang": "en"})


def test_package(nlp: Language, chdir):
    d = Path().cwd()
    modeld = d / "model"
    pkgd = d / "package"
    pkgd.mkdir()
    nlp.to_disk(modeld)
    package(modeld, pkgd)
