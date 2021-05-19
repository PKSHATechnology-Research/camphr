from collections import defaultdict
import dataclasses
import logging
import os
from pathlib import Path
import random
from typing import Any, Callable, Dict, Tuple, Type, Union, cast

import dataclass_utils
import numpy as np
from spacy.language import Language
from spacy.util import minibatch
import torch
from torch.optim.optimizer import Optimizer
import yaml

from camphr.models import create_model
from camphr.pipelines.transformers.seq_classification import TOP_LABEL
from camphr_cli.config import TrainConfig
from camphr_cli.utils import (
    InputData,
    check_nonempty,
    convert_fullpath_if_path,
    create_data,
    report_fail,
    unzip2,
)
from camphr_core.lang.torch import TorchLanguage
from camphr_core.torch_utils import goldcat_to_label
from camphr_core.utils import (
    create_dict_from_dotkey,
    get_by_dotkey,
    import_attr,
    merge_dicts,
    resolve_alias,
)
import hydra
import hydra.utils
from omegaconf import Config, OmegaConf
from sklearn.metrics import classification_report

logger = logging.getLogger(__name__)


MUST_FIELDS = [
    "train.data.path",
    [
        "model.ner_label",
        "model.textcat_label",
        "model.multitextcat_label",
        "model.pipeline.transformers_ner.labels",
        "model.pipeline.transformers_seq_classification.labels",
        "model.pipeline.transformers_multilabel_seq_classification.labels",
        "model.labels",
        "model.task",
    ],
    "model.lang.name",
]

PATH_FIELDS = [
    "model.ner_label",
    "model.textcat_label",
    "model.multitextcat_label",
    "model.labels",
    "train.data.path",
    "model.pretrained",
]

ALIASES = {
    "train.optimizer": "model.lang.optimizer",
    "data": "train.data",
    "lang": "model.lang.name",
}


def resolve_path(cfg: Dict[str, Any]) -> Dict[str, Any]:
    for key in PATH_FIELDS:
        path = get_by_dotkey(cfg, key)
        if path:
            path = convert_fullpath_if_path(path)
            cfg = merge_dicts(cfg, create_dict_from_dotkey(key, path))
    return cfg


def parse(cfg: TrainConfig) -> TrainConfig:
    cfg_dict = dataclasses.asdict(cfg)
    cfg_dict = resolve_alias(ALIASES, cfg_dict)
    check_nonempty(cfg_dict, MUST_FIELDS)
    cfg_dict = resolve_path(cfg_dict)
    return dataclass_utils.into(cfg_dict, TrainConfig)


def evaluate_textcat(cfg: Config, nlp: TorchLanguage, val_data: InputData) -> Dict:
    # TODO: https://github.com/explosion/spaCy/pull/4664
    texts, golds = cast(Tuple[Tuple[str], Dict], zip(*val_data))
    try:
        y = np.array(list(map(lambda x: goldcat_to_label(x["cats"]), golds)))
        docs = list(nlp.pipe(texts, batch_size=cfg.nbatch * 2))
        preds = np.array([doc._.get(TOP_LABEL) for doc in docs])
    except Exception:
        report_fail(val_data)
        raise
    return classification_report(y, preds, output_dict=True)


def evaluate(cfg: TrainConfig, nlp: TorchLanguage, val_data: InputData) -> Dict:
    try:
        scores = nlp.evaluate(val_data, batch_size=cfg.train.nbatch * 2)
    except Exception:
        report_fail(val_data)
        raise
    return scores


EvalFn = Callable[[Config, TorchLanguage, InputData], Dict]

EVAL_FN_MAP = defaultdict(  # type: ignore
    lambda: evaluate, {"textcat": evaluate_textcat}  # type: ignore
)


def train_epoch(
    cfg: TrainConfig,
    nlp: TorchLanguage,
    optim: Optimizer,
    train_data: InputData,
    val_data: InputData,
    epoch: int,
    eval_fn: EvalFn,
) -> None:
    for j, batch in enumerate(minibatch(train_data, size=cfg.train.nbatch)):
        texts, golds = unzip2(batch)
        try:
            nlp.update(texts, golds, optim, verbose=True)
        except Exception:
            report_fail(batch)
            raise
        logger.info(f"epoch {epoch} {j*cfg.train.nbatch}/{cfg.train.data.ndata}")


def save_model(nlp: Language, path: Path) -> None:
    nlp.to_disk(path)
    logger.info(f"Saved the model in {str(path.absolute())}")


class DummyScheduler:
    @staticmethod
    def step() -> None:
        ...


def load_scheduler(
    cfg: TrainConfig, optimizer: Optimizer
) -> Union[torch.optim.lr_scheduler.LambdaLR, Type[DummyScheduler]]:
    if not cfg.train.scheduler or not cfg.train.scheduler.class_:
        return DummyScheduler
    cls_str = cfg.train.scheduler.class_
    cls = cast(Type[torch.optim.lr_scheduler.LambdaLR], import_attr(cls_str))
    params = cfg.train.scheduler.params or {}
    return cls(optimizer, **params)


def train(
    cfg: TrainConfig,
    nlp: TorchLanguage,
    train_data: InputData,
    val_data: InputData,
    savedir: Path,
) -> None:
    task = cfg.model.task
    if task is None:
        raise ValueError("Task is not specified.")
    eval_fn = EVAL_FN_MAP[task]
    optim = nlp.resume_training()
    scheduler = load_scheduler(cfg, optim)
    for i in range(cfg.train.niter):
        random.shuffle(train_data)
        train_epoch(cfg, nlp, optim, train_data, val_data, i, eval_fn)
        scheduler.step()  # type: ignore # (https://github.com/pytorch/pytorch/pull/26531)
        scores = eval_fn(cfg, nlp, val_data)
        nlp.meta.update({"score": scores, "config": dataclasses.asdict(cfg)})
        save_model(nlp, savedir / str(i))


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # type: ignore


def validate_data(cfg: Config, data: InputData, n_check=100):
    if "textcat" in cfg.model.pipeline:
        data = random.sample(data, min(n_check, len(data)))
        for text, gold in data:
            assert "cats" in gold, "`cats` key is required in gold label"
            assert (
                abs(sum(gold["cats"].values()) - 1) < 1e-2
            ), "Sum of gold.cats must equal 1. for text classification task."


def _main(cfg: Config) -> None:
    if cfg.get("user_config") is not None:
        # Override config by user config.
        # This `user_config` have some limitations, and it will be improved
        # after the issue https://github.com/facebookresearch/hydra/issues/386 solved
        cfg = OmegaConf.merge(
            cfg, OmegaConf.load(hydra.utils.to_absolute_path(cfg["user_config"]))
        )
    if isinstance(cfg, Config):
        cfg_dict = cfg.to_container()
    else:
        cfg_dict = cfg
    cfg_ = dataclass_utils.into(cfg_dict, TrainConfig)
    cfg_ = parse(cfg_)
    if cfg_.seed:
        set_seed(cfg_.seed)
    logger.info(yaml.dump(cfg))
    nlp = cast(TorchLanguage, create_model(cfg_.model))
    train_data, val_data = create_data(cfg_.train.data)
    validate_data(cfg_, train_data)
    logger.info("output dir: {}".format(os.getcwd()))
    if torch.cuda.is_available():
        logger.info("CUDA enabled")
        nlp.to(torch.device("cuda"))
    savedir = Path.cwd() / "models"
    savedir.mkdir(exist_ok=True)
    train(cfg_, nlp, train_data, val_data, savedir)


# Avoid to use decorator for testing
main = hydra.main(config_path="conf/train/config.yaml", strict=False)(_main)

if __name__ == "__main__":
    main()
