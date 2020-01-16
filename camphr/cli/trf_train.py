import logging
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict

import hydra
import hydra.utils
import numpy as np
import omegaconf
import torch
from camphr.cli.utils import InputData, create_data, evaluate, report_fail, validate
from camphr.lang.torch import TorchLanguage
from camphr.models import create_model
from camphr.pipelines.trf_seq_classification import TOP_LABEL
from camphr.torch_utils import goldcat_to_label
from sklearn.metrics import classification_report
from spacy.language import Language
from spacy.util import minibatch

log = logging.getLogger(__name__)


def evaluate_textcat(cfg: omegaconf.Config, nlp: Language, val_data) -> Dict:
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


EvalFn = Callable[[omegaconf.Config, Language, InputData], Dict]

EVAL_FN_MAP = defaultdict(lambda: evaluate, {"textcat": evaluate_textcat})


def train_epoch(
    cfg: omegaconf.Config,
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
            nlp.update(texts, golds, optim, debug=True)
        except Exception:
            report_fail(batch)
            raise
        log.info(f"epoch {epoch} {j*cfg.nbatch}/{cfg.data.ndata}")


def save_model(nlp: Language, path: Path) -> None:
    nlp.to_disk(path)
    log.info(f"Saved the model in {str(path.absolute())}")


def train(
    cfg: omegaconf.Config,
    nlp: TorchLanguage,
    train_data: InputData,
    val_data: InputData,
    savedir: Path,
) -> None:
    eval_fn = EVAL_FN_MAP[cfg.task]
    optim = nlp.resume_training(t_total=cfg.niter)
    for i in range(cfg.niter):
        random.shuffle(train_data)
        train_epoch(cfg, nlp, optim, train_data, val_data, i, eval_fn)
        scores = eval_fn(cfg, nlp, val_data)
        nlp.meta.update(
            {"score": scores, "config": omegaconf.OmegaConf.to_container(cfg)}
        )
        save_model(nlp, savedir / str(i))


def _main(cfg: omegaconf.Config) -> None:
    cfg = validate(cfg)
    log.info(cfg.pretty())
    train_data, val_data = create_data(cfg.train.data)
    nlp = create_model(cfg.model)
    log.info("output dir: {}".format(os.getcwd()))
    if torch.cuda.is_available():
        log.info("CUDA enabled")
        nlp.to(torch.device("cuda"))
    savedir = Path.cwd() / "models"
    savedir.mkdir(exist_ok=True)
    train(cfg.train, nlp, train_data, val_data, savedir)


# Avoid to use decorator for testing
main = hydra.main(config_path="conf/trf_train/config.yaml", strict=False)(_main)

if __name__ == "__main__":
    main()
