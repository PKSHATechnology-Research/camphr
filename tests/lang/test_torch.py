import pytest
import spacy
import torch
from camphr.lang.torch import TorchLanguage, get_torch_nlp


@pytest.mark.parametrize("lang", ["en", "ja_mecab", "fa"])
def test_torchlang(lang, tmp_path):
    nlp = get_torch_nlp(lang)
    assert isinstance(nlp, TorchLanguage)
    assert nlp.lang == lang
    nlp.to(torch.device("cpu"))
    nlp.to_disk(tmp_path)
    nlp = spacy.load(tmp_path)
    assert isinstance(nlp, TorchLanguage)
