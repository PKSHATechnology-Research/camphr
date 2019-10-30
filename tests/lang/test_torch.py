import spacy
import torch
import pytest
import bedoner.lang.mecab as mecab
import bedoner.lang.juman as juman
import bedoner.lang.knp as knp
import bedoner.lang.sentencepiece as sp


@pytest.mark.parametrize(
    "cls",
    [
        mecab.TorchJapanese,
        juman.TorchJapanese,
        knp.TorchJapanese,
        sp.TorchSentencePieceLang,
    ],
)
def test_torchlang_serialization(cls, tmpdir):
    nlp = cls()
    nlp.to(torch.device("cpu"))
    nlp.to_disk(str(tmpdir))
    nlp = spacy.load(str(tmpdir))
    assert isinstance(nlp, cls)
    nlp.to(torch.device("cpu"))
