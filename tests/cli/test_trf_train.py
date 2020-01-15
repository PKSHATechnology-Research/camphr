from pathlib import Path

import omegaconf
import pytest
from camphr.cli.trf_train import _main
from camphr.pipelines.trf_ner import TRANSFORMERS_NER

from ..utils import BERT_DIR

DATA_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(
    params=[
        (
            "ner.yml",
            f"""
            model:
                pipeline:
                    {TRANSFORMERS_NER}:
                        trf_name_or_path: {BERT_DIR}
                        labels: {DATA_DIR/"irex.json"}
            train:
                data:
                    path: {DATA_DIR / "test_ner_irex_ja.jsonl"}
            """,
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
