import logging
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List

import hydra
import numpy as np
import omegaconf
import srsly
import torch
from camphr.cli.utils import InputData, create_data, evaluate, report_fail
from camphr.lang.torch_mixin import TorchLanguage
from camphr.models import get_trf_name, trf_ner, trf_seq_classification
from camphr.ner_labels.utils import make_biluo_labels
from camphr.pipelines.trf_seq_classification import TOP_LABEL
from camphr.torch_utils import goldcat_to_label
from sklearn.metrics import classification_report
from spacy.language import Language
from spacy.util import minibatch

log = logging.getLogger(__name__)


def get_labels(cfg: omegaconf.Config) -> List[str]:
    if cfg.path:
        labels = srsly.read_json(Path(cfg.path).expanduser())
        return make_biluo_labels(labels)
    return []


def create_nlp(cfg: omegaconf.Config) -> TorchLanguage:
    labels = get_labels(cfg.label)
    nlp = None
    if cfg.task == "ner":
        nlp = trf_ner(
            lang=cfg.lang, pretrained=cfg.pretrained, labels=labels, freeze=cfg.freeze
        )
    elif cfg.task == "textcat":
        nlp = trf_seq_classification(
            lang=cfg.lang, pretrained=cfg.pretrained, labels=labels, freeze=cfg.freeze
        )
    else:
        raise ValueError(f"Cannot use \"{cfg.task}\" as 'cfg.task'.")
    name = get_trf_name(cfg.pretrained)
    nlp.meta["name"] = name.value + "_" + (cfg.model_name or cfg.task)
    return nlp


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
        log.info(f"epoch {epoch} {j*cfg.nbatch}/{cfg.ndata}")


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
    log.info(cfg.pretty())
    log.info("output dir: {}".format(os.getcwd()))
    train_data, val_data = create_data(cfg.train.data)
    nlp = create_nlp(cfg.model)
    if torch.cuda.is_available():
        log.info("CUDA enabled")
        nlp.to(torch.device("cuda"))
    savedir = Path.cwd() / "models"
    savedir.mkdir(exist_ok=True)
    train(cfg.train, nlp, train_data, val_data, savedir)


# Avoid to use decorator because of testing.
main = hydra.main(config_path="conf/train.yml")(_main)

if __name__ == "__main__":
    main()
