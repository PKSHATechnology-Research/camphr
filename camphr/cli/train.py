import logging
import optuna
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, Type, Union

import hydra
import hydra.utils
import numpy as np
import torch
from omegaconf import Config, OmegaConf
from sklearn.metrics import classification_report
from spacy.language import Language
from spacy.scorer import Scorer
from spacy.util import minibatch

from camphr.cli.utils import (
    InputData,
    check_nonempty,
    convert_fullpath_if_path,
    create_data,
    report_fail,
)
from camphr.lang.torch import TorchLanguage
from camphr.models import correct_model_config, create_model
from camphr.pipelines.transformers.seq_classification import TOP_LABEL
from camphr.torch_utils import goldcat_to_label
from camphr.utils import (
    create_dict_from_dotkey,
    get_by_dotkey,
    import_attr,
    resolve_alias,
)

log = logging.getLogger(__name__)


MUST_FIELDS = [
    "train.data.path",
    [
        "model.ner_label",
        "model.textcat_label",
        "model.pipeline.transformers_ner.labels",
        "model.pipeline.transformers_seq_classification.labels",
    ],
    "model.lang.name",
]

PATH_FIELDS = [
    "model.ner_label",
    "model.textcat_label",
    "train.data.path",
    "model.pretrained",
]

ALIASES = {"train.optimizer": "model.lang.optimizer"}


def resolve_path(cfg: Config) -> Config:
    for key in PATH_FIELDS:
        path = get_by_dotkey(cfg, key)
        if path:
            path = convert_fullpath_if_path(path)
            cfg = OmegaConf.merge(
                cfg, OmegaConf.create(create_dict_from_dotkey(key, path))
            )
    return cfg


def parse(cfg: Config):
    check_nonempty(cfg, MUST_FIELDS)
    cfg = resolve_path(cfg)
    cfg = resolve_alias(ALIASES, cfg)
    cfg.model = correct_model_config(cfg.model)
    return cfg


def evaluate_textcat(cfg: Config, nlp: Language, val_data) -> Dict:
    # TODO: https://github.com/explosion/spaCy/pull/4664
    texts, golds = zip(*val_data)
    try:
        y = np.array(list(map(lambda x: goldcat_to_label(x["cats"]), golds)))
        docs = list(nlp.pipe(texts, batch_size=cfg.nbatch * 2))
        preds = np.array([doc._.get(TOP_LABEL) for doc in docs])
    except Exception:
        report_fail(val_data)
        raise
    return classification_report(y, preds, output_dict=True)


def evaluate(cfg: Config, nlp: Language, val_data: InputData) -> Dict:
    try:
        scorer: Scorer = nlp.evaluate(val_data, batch_size=cfg.nbatch * 2)
    except Exception:
        report_fail(val_data)
        raise
    return scorer.scores


EvalFn = Callable[[Config, Language, InputData], Dict]

EVAL_FN_MAP = defaultdict(lambda: evaluate, {"textcat": evaluate_textcat})


def train_epoch(
    cfg: Config,
    nlp: TorchLanguage,
    optim: torch.optim.Optimizer,
    train_data: InputData,
    val_data: InputData,
    epoch: int,
    eval_fn: EvalFn,
) -> None:
    for j, batch in enumerate(minibatch(train_data, size=cfg.nbatch)):
        texts, golds = zip(*batch)
        try:
            nlp.update(texts, golds, optim, verbose=True)
        except Exception:
            report_fail(batch)
            raise
        log.info(f"epoch {epoch} {j*cfg.nbatch}/{cfg.data.ndata}")


def save_model(nlp: Language, path: Path) -> None:
    nlp.to_disk(path)
    log.info(f"Saved the model in {str(path.absolute())}")


class DummyScheduler:
    @staticmethod
    def step():
        ...


def load_scheduler(
    cfg: Config, optimizer: torch.optim.Optimizer
) -> Union[torch.optim.lr_scheduler.LambdaLR, Type[DummyScheduler]]:
    cls_str = get_by_dotkey(cfg, "scheduler.class")
    if not cls_str:
        return DummyScheduler
    cls = import_attr(cls_str)
    params = OmegaConf.to_container(cfg.scheduler.params) or {}
    return cls(optimizer, **params)


def train(
    cfg: Config,
    nlp: TorchLanguage,
    train_data: InputData,
    val_data: InputData,
    savedir: Path,
) -> float:
    eval_fn = EVAL_FN_MAP[cfg.task]
    optim = nlp.resume_training()
    scheduler = load_scheduler(cfg, optim)
    scores = None
    for i in range(cfg.niter):
        random.shuffle(train_data)
        train_epoch(cfg, nlp, optim, train_data, val_data, i, eval_fn)
        scheduler.step()  # noqa: invalid type annotation in pytorch
        scores = eval_fn(cfg, nlp, val_data)
        nlp.meta.update({"score": scores, "config": OmegaConf.to_container(cfg)})
        save_model(nlp, savedir / str(i))
    return -scores["ents_f"] if scores else 1e5


def _main(cfg: Config) -> None:
    if cfg.user_config is not None:
        # Override config by user config.
        # This `user_config` have some limitations, and it will be improved
        # after the issue https://github.com/facebookresearch/hydra/issues/386 solved
        cfg = OmegaConf.merge(
            cfg, OmegaConf.load(hydra.utils.to_absolute_path(cfg.user_config))
        )
    cfg = parse(cfg)
    log.info(cfg.pretty())
    train_data, val_data = create_data(cfg.train.data)

    def objective(trial: optuna.Trial):
        savedir = Path.cwd() / f"models{trial.trial_id}"
        savedir.mkdir(exist_ok=True)
        cfg.model.lang.optimizer.params.lr = trial.suggest_uniform("lr", 1e-7, 1e-1)
        cfg.model.lang.optimizer.params.eps = trial.suggest_uniform("eps", 1e-10, 1e-2)
        nlp = create_model(cfg.model)
        if torch.cuda.is_available():
            log.info("CUDA enabled")
            nlp.to(torch.device("cuda"))
        return train(cfg.train, nlp, train_data, val_data, savedir)

    study = optuna.create_study(study_name="capmhr", storage="sqlite:///optuna.db")
    study.optimize(objective, n_trials=cfg.optuna.ntrials)
    log.info(f"Best trial: {study.best_trial.trial_id}")
    log.info(f"Best score: {study.best_value}")


# Avoid to use decorator for testing
main = hydra.main(config_path="conf/train/config.yaml", strict=False)(_main)

if __name__ == "__main__":
    main()
