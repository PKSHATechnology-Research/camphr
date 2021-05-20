from typing import Any, Dict

from camphr_core.utils import yaml_to_dict
from camphr_test.utils import check_lang, check_mecab
from camphr_torch.lang import TorchLanguage
import pytest
import spacy
from spacy.vocab import Vocab
import torch

from camphr_transformers.model import TRANSFORMERS_MODEL, TrfModel
from camphr_transformers.tokenizer import TrfTokenizer

from .utils import TRF_TESTMODEL_PATH

ALL_LANGS = ["ja_mecab", "ja_juman"]


@pytest.fixture(scope="session", params=ALL_LANGS)
def lang(request):
    if not check_lang(request.param):
        pytest.skip(f"No requirements for {request.param}")
    return request.param


@pytest.fixture(scope="session", params=["cuda", "cpu"])
def device(request):
    if request.param == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        pytest.skip("cuda is required")
    return torch.device("cuda")


@pytest.fixture(scope="session", params=TRF_TESTMODEL_PATH)
def trf_name_or_path(request):
    name = request.param
    if "bert-base-japanese" in name and not check_mecab():
        pytest.skip("mecab is required")
    return name


@pytest.fixture(scope="module")
def nlp_trf_model(lang: str, trf_name_or_path: str, device):
    _nlp = TorchLanguage(
        meta={"lang": lang},
        optimizer_config={"class": "torch.optim.SGD", "params": {"lr": 0.01}},
    )
    _nlp.add_pipe(TrfTokenizer.from_pretrained(_nlp.vocab, trf_name_or_path))
    _nlp.add_pipe(TrfModel.from_pretrained(_nlp.vocab, trf_name_or_path))
    _nlp.to(device)
    return _nlp
