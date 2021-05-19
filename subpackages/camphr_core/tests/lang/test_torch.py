from camphr_test.utils import check_mecab
import pytest
import spacy
import torch

from camphr_core.lang.torch import TorchLanguage, get_torch_nlp


@pytest.fixture
def dummy_params():
    return torch.arange(10).float()


@pytest.fixture(
    params=[
        {"class": "torch.optim.SGD", "params": {"lr": 0.01}},
        {"class": "torch.optim.Adam", "params": {"lr": 1}},
    ]
)
def optimizer_config(request):
    return request.param


@pytest.mark.parametrize("lang", ["en", "ja_mecab", "fa"])
def test_torchlang(lang, tmp_path, device, dummy_params, optimizer_config):
    if lang == "ja_mecab" and not check_mecab():
        pytest.skip()
    nlp = get_torch_nlp(lang, optimizer_config=optimizer_config)
    assert isinstance(nlp, TorchLanguage)
    assert nlp.lang == lang
    optim = nlp.create_optimizer([dummy_params])
    assert optim.__class__.__name__ == optimizer_config["class"].split(".")[-1]
    nlp.to(device)
    nlp.to_disk(tmp_path)
    nlp = spacy.load(tmp_path)
    assert isinstance(nlp, TorchLanguage)
