import os
from pathlib import Path

import omegaconf
import pytest
from camphr.cli.trf_train import _main

from ..utils import BERT_DIR


@pytest.fixture
def chdir(tmp_path: Path):
    tmp_path.mkdir(exist_ok=True)
    cwd = os.getcwd()
    os.chdir(tmp_path)
    yield
    os.chdir(cwd)


DATA_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(
    params=[
        (
            "ner.yml",
            {
                "model": {
                    "pretrained": str(BERT_DIR),
                    "label": {"path": str(DATA_DIR / "irex.json")},
                },
                "train": {"data": {"path": str(DATA_DIR / "test_ner_irex_ja.jsonl")}},
            },
        )
    ]
)
def config(request):
    path, diff = request.param
    _config = omegaconf.OmegaConf.load(
        str(Path(__file__).parent / "fixtures/confs/trf_train" / path)
    )
    diff = omegaconf.OmegaConf.create(diff)
    _config = omegaconf.OmegaConf.merge(_config, diff)
    return _config


def test_main(config, chdir):
    _main(config)
