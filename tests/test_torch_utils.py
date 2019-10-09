from typing import List

import pytest
import torch
import torch.nn as nn
from bedoner.lang.torch_mixin import TorchLanguageMixin
from bedoner.torch_utils import TorchPipe
from spacy.language import Language
from spacy.tokens import Doc


class TorchLang(TorchLanguageMixin, Language):
    pass


@pytest.fixture
def torch_lang():
    return TorchLang()


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


def test_torch_language(torch_nlp: TorchLang):
    doc = torch_nlp("This is an apple")
    assert doc._.score
