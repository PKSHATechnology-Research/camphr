from typing import Any, Dict

from camphr_core.utils import yaml_to_dict
from camphr_test.utils import check_mecab

from camphr_pipelines.models import create_model
from camphr_transformers.model import TRANSFORMERS_MODEL
import pytest
import torch

from .utils import TRF_TESTMODEL_PATH


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


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


@pytest.fixture(scope="session")
def trf_model_config(lang, trf_name_or_path, device) -> Dict[str, Any]:
    return yaml_to_dict(
        f"""
    lang:
        name: {lang}
        optimizer:
            class: torch.optim.SGD
            params:
                lr: 0.01
    pipeline:
        {TRANSFORMERS_MODEL}:
          trf_name_or_path: {trf_name_or_path}
    """
    )


@pytest.fixture(scope="module")
def nlp_trf_model(lang, trf_name_or_path, device):
    _nlp = create_model(trf_model_config)
    _nlp.to(device)
    return _nlp
