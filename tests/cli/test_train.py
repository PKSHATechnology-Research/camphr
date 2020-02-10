from pathlib import Path

import pytest
from omegaconf import Config, OmegaConf

from camphr.cli.train import _main, set_seed
from camphr.models import create_model
from camphr.pipelines.transformers.ner import TRANSFORMERS_NER

from ..utils import BERT_DIR, XLNET_DIR

DATA_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def default_config() -> Config:
    return OmegaConf.load(
        str(
            Path(__file__).parent.parent.parent
            / "camphr"
            / "cli"
            / "conf"
            / "train"
            / "config.yaml"
        )
    )


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
                niter: 1
            """,
        )
    ]
)
def config(request, default_config):
    ident, diff = request.param
    diff = OmegaConf.create(diff)
    _config = OmegaConf.merge(default_config, diff)
    return _config


def test_main(config, chdir):
    _main(config)


def test_seed(chdir, default_config):
    cfg = f"""
    model:
        lang:
            name: en
        pipeline:
            {TRANSFORMERS_NER}:
                trf_name_or_path: {XLNET_DIR}
                labels: {DATA_DIR/"irex.json"}
    seed: 0
    """

    def get_model_value(cfg):
        nlp = create_model(cfg.model)
        pipe = nlp.get_pipe(TRANSFORMERS_NER)
        return sum(p.sum().cpu().item() for p in pipe.model.parameters())

    cfg = OmegaConf.merge(default_config, OmegaConf.create(cfg))
    set_seed(cfg.seed)
    first = get_model_value(cfg)
    set_seed(cfg.seed)
    second = get_model_value(cfg)
    assert first == second
