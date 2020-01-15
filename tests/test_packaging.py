from pathlib import Path

import pytest
from camphr.lang.torch import TorchLanguage
from spacy.cli import package
from spacy.language import Language


@pytest.fixture(scope="session")
def nlp():
    return TorchLanguage(meta={"lang": "ja_mecab"})


def test_package(nlp: Language, chdir):
    d = Path().cwd()
    modeld = d / "model"
    pkgd = d / "package"
    pkgd.mkdir()
    nlp.to_disk(modeld)
    package(modeld, pkgd)
