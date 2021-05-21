from typing import Any, Dict

from camphr_test.utils import check_lang
import dataclass_utils

from camphr_pipelines.models import NLPConfig, create_model, load
from camphr_transformers.model import TRANSFORMERS_MODEL
import pytest


def test_freeze_model(trf_name_or_path, trf_model_config: Dict[str, Any]):
    config = dataclass_utils.into(trf_model_config, NLPConfig)
    config.pipeline[TRANSFORMERS_MODEL]["freeze"] = True
    nlp = create_model(config)
    pipe = nlp.pipeline[-1][1]
    assert pipe.cfg["freeze"]


@pytest.mark.slow
@pytest.mark.parametrize("name", ["bert-base-cased", "xlm-roberta-base"])
def test_load_transformers(name):
    cfg = f"""
    lang:
        name: en
        optimizer: {{}}
    pipeline:
        transformers_model:
            trf_name_or_path: {name}
    """
    load(cfg)


ALL_LANGS = ["ja_mecab", "ja_juman"]


@pytest.fixture(scope="session", params=ALL_LANGS)
def lang(request):
    if not check_lang(request.param):
        pytest.skip(f"No requirements for {request.param}")
    return request.param
