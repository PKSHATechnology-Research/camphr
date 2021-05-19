import json
from pathlib import Path

import pytest
import torch

from camphr.ner_labels.labels_ene import ALL_LABELS
from camphr_core.lang.torch import TorchLanguage
from camphr_pipelines.models import create_model


@pytest.fixture
def nlp():
    name = "albert-base-v2"
    config = f"""
    lang:
        name: en
        optimizer:
            class: torch.optim.SGD
            params:
                lr: 0.01
    pipeline:
        transformers_model:
          trf_name_or_path: {name}
        transformers_ner:
          labels: {ALL_LABELS}
    """
    return create_model(config)


@pytest.fixture
def data():
    return json.loads((Path(__file__).parent / "fail.json").read_text())


def test(nlp: TorchLanguage, data):
    if torch.cuda.is_available():
        nlp.to(torch.device("cuda"))
    nlp.evaluate(data, batch_size=256)
