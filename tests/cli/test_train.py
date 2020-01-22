from pathlib import Path

import pytest
from omegaconf import OmegaConf

from camphr.cli.train import _main
from camphr.pipelines.transformers.ner import TRANSFORMERS_NER

from ..utils import BERT_DIR

DATA_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(
    params=[
        (
            "foo",
            f"""
            model:
                lang:
                    name: ja
                pipeline:
                    {TRANSFORMERS_NER}:
                        trf_name_or_path: {BERT_DIR}
                        labels: {DATA_DIR/"irex.json"}
            train:
                data:
                    path: {DATA_DIR / "test_ner_irex_ja.jsonl"}
            """,
        ),
        (
            "batch failure",
            f"""
            model:
                lang:
                    name: ja
                ner_label: {DATA_DIR/"irex.json"}
                pretrained: bert-base-japanese
            train:
                data:
                    path: {DATA_DIR / "test_ner_irex_ja.jsonl"}
            task: ner
            """,
        ),
    ]
)
def config(request):
    ident, diff = request.param
    _config = OmegaConf.load(
        str(
            Path(__file__).parent.parent.parent
            / "camphr"
            / "cli"
            / "conf"
            / "train"
            / "config.yaml"
        )
    )
    diff = OmegaConf.create(diff)
    _config = OmegaConf.merge(_config, diff)
    return _config


@pytest.mark.slow
def test_main(config, chdir):
    _main(config)
