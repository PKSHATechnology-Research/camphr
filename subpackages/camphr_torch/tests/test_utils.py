from pathlib import Path
from typing import List

import pytest
import spacy
from spacy.tokens import Doc
from spacy.vocab import Vocab
import torch
import torch.nn as nn

from camphr_torch.lang import TorchLanguage
from camphr_torch.utils import TensorWrapper, TorchPipe


@pytest.fixture
def torch_lang():
    return TorchLanguage()


class SamplePipe(TorchPipe):
    def __init__(self):
        Doc.set_extension("score", default=None)
        self.insize = 1
        self.outsize = 3
        self.model = nn.Linear(self.insize, self.outsize)

    def predict(self, docs: List[Doc]):
        scores = torch.stack(
            [self.model(torch.tensor([float(len(doc))])) for doc in docs]
        )
        return scores

    def set_annotations(self, docs, scores):
        for doc, score in zip(docs, scores):
            doc._.score = score.sum()


@pytest.fixture
def torch_pipe():
    return SamplePipe()


@pytest.fixture
def torch_nlp(torch_pipe, torch_lang):
    torch_lang.add_pipe(torch_pipe)
    return torch_lang


def test_torch_language(torch_nlp: TorchLanguage):
    doc = torch_nlp("This is an apple")
    assert doc._.score


@pytest.mark.skip(
    reason="""How to store the doc?
- https://stackoverflow.com/questions/59744811/how-to-store-custom-class-object-into-spacy-doc-and-use-doc-to-disk
- https://github.com/explosion/srsly/issues/20
"""
)
def test_tensorwrapper(tmp_path: Path):
    doc = Doc(Vocab(), words=["This", "is", "a", "test"])
    doc.user_data["tensor"] = TensorWrapper(torch.zeros((2, 2)), 1)
    doc.to_disk(tmp_path / "doc.bin")
    spacy.load(tmp_path / "doc.bin")
