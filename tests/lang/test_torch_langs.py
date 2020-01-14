import pytest
import spacy
import torch
from camphr.lang.torch_langs import get_torch_lang_cls
from camphr.lang.torch_mixin import TorchLanguageMixin
from spacy.util import get_lang_class


@pytest.mark.parametrize("lang", ["en", "ja_mecab", "fa"])
def test_torchlang(lang, tmp_path):
    cls = get_lang_class(lang)
    torchcls = get_torch_lang_cls(lang)
    assert issubclass(torchcls, TorchLanguageMixin)
    assert issubclass(torchcls, cls)
    assert torchcls.lang == lang + "_torch"
    nlp = torchcls()
    nlp.to(torch.device("cpu"))
    nlp.to_disk(tmp_path)
    nlp = spacy.load(tmp_path)
    assert isinstance(nlp, torchcls)
