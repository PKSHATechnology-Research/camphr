from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict

import dataclass_utils
import pytest
import yaml

from camphr import __version__
from camphr.models import create_model
from camphr.pipelines.transformers.ner import TRANSFORMERS_NER
from camphr_cli.config import TrainConfig
from camphr_cli.train import _main, set_seed, validate_data
from camphr_core.utils import merge_dicts
from omegaconf import OmegaConf

from .utils import BERT_DIR, BERT_JA_DIR, FIXTURE_DIR, XLNET_DIR, check_mecab


@pytest.fixture
def default_config() -> Dict[str, Any]:
    return yaml.safe_load(
        Path(
            Path(__file__).parent.parent
            / "camphr_cli"
            / "conf"
            / "train"
            / "config.yaml"
        ).read_text()
    )


@pytest.fixture(
    params=[
        (
            "ner",
            f"""
            model:
                lang:
                    name: ja_mecab
                task: ner
                pretrained: {BERT_JA_DIR}
                labels: {FIXTURE_DIR/"irex.json"}
            train:
                data:
                    path: {FIXTURE_DIR / "test_ner_irex_ja.jsonl"}
                niter: 1
            """,
            not check_mecab(),
        ),
        (
            "multitext",
            f"""
            model:
                lang:
                    name: en
                pretrained: {BERT_DIR}
                task: multilabel_textcat
                labels: {FIXTURE_DIR/"multi-textcat"/"label.json"}
            train:
                data:
                    path: {FIXTURE_DIR / "multi-textcat"/ "train.jsonl"}
                niter: 1
            """,
            (__version__ <= "0.5.3"),
        ),
    ]
)
def config(request, default_config):
    ident, diff, skip = request.param
    if skip:
        pytest.skip()
    diff = yaml.safe_load(diff)
    _config = merge_dicts(default_config, diff)
    return _config


def test_main(config, chdir):
    _main(config)


def test_cli(config: Dict[str, Any], chdir):
    cfgpath = Path("user.yaml").absolute()
    cfgpath.write_text(json.dumps(config))
    res = subprocess.run(
        [sys.executable, "-m", "camphr_cli", "train", f"user_config={cfgpath}"],
        stderr=subprocess.PIPE,
    )
    assert res.returncode == 0, res.stderr.decode()


def test_seed(chdir, default_config):
    cfg = f"""
    model:
        lang:
            name: en
        pipeline:
            {TRANSFORMERS_NER}:
                trf_name_or_path: {XLNET_DIR}
                labels: {FIXTURE_DIR/"irex.json"}
    seed: 0
    """

    def get_model_value(cfg: TrainConfig):
        nlp = create_model(cfg.model)
        pipe = nlp.get_pipe(TRANSFORMERS_NER)
        return sum(p.sum().cpu().item() for p in pipe.model.parameters())

    cfg_dict = merge_dicts(default_config, yaml.safe_load(cfg))
    cfg = dataclass_utils.into(cfg_dict, TrainConfig)
    assert cfg.seed is not None
    set_seed(cfg.seed)
    first = get_model_value(cfg)
    set_seed(cfg.seed)
    second = get_model_value(cfg)
    assert first == second


@contextmanager
def does_not_raise():
    yield


@pytest.mark.parametrize(
    "cfg,data,raises",
    [
        (
            """
    model:
        pipeline:
            textcat:
    """,
            [("foo", {"cats": {"FOO": 1.0, "BAR": 0.0}})],
            does_not_raise(),
        ),
        (
            """
    model:
        pipeline:
            textcat:
    """,
            [("foo", {"cats": {"FOO": 1.0, "BAR": 0.3}})],
            pytest.raises(AssertionError),
        ),
    ],
)
def test_validate_data(cfg, data, raises):
    with raises:
        validate_data(OmegaConf.create(cfg), data)
