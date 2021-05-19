from typing import Any, Dict

from camphr_pipelines.models import NLPConfig, create_model, load
import dataclass_utils
import pytest

from camphr_transformers.model import TRANSFORMERS_MODEL


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
    pipeline:
        transformers_model:
            trf_name_or_path: {name}
    """
    load(cfg)
