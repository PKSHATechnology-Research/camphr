import os
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple

import hydra
from spacy.language import Language
import omegaconf
import torch
from sklearn.model_selection import train_test_split
from spacy.scorer import Scorer
from spacy.util import minibatch

from bedoner.models import trf_ner, get_trf_name
from bedoner.ner_labels import LABELS
from bedoner.ner_labels.utils import make_biluo_labels

log = logging.getLogger(__name__)


def load_data(name: str) -> List[Dict]:
    name = os.path.expanduser(name)
    data = []
    with open(name) as f:
        for line in f:
            data.append(json.loads(line))
    return data


class Config(omegaconf.OmegaConf):
    data: str
    ndata: int
    niter: int
    nbatch: int
    label: str
    scheduler: bool
    test_size: float
    lang: str
    pretrained: str
    neval: int


def create_data(cfg: Config) -> Tuple[List, List]:
    data = load_data(cfg.data)
    if cfg.ndata != -1:
        data = random.sample(data, k=cfg.ndata)
    else:
        cfg.ndata = len(data)
    return train_test_split(data, test_size=cfg.test_size)


def create_nlp(cfg: Config) -> Language:
    labels = LABELS[cfg.label]
    nlp = trf_ner(
        lang=cfg.lang, pretrained=cfg.pretrained, labels=make_biluo_labels(labels)
    )
    name = get_trf_name(cfg.pretrained)
    nlp.meta["name"] = name.value + "_" + cfg.label
    return nlp


def evaluate(cfg: Config, nlp, val_data) -> Scorer:
    try:
        scorer: Scorer = nlp.evaluate(val_data, batch_size=cfg.nbatch * 2)
    except:
        with open("fail.json", "w") as f:
            json.dump(val_data, f, ensure_ascii=False)
            raise
    return scorer


def train_epoch(cfg: Config, nlp, optim, train_data, val_data, epoch, eval_fn):
    for j, batch in enumerate(minibatch(train_data, size=cfg.nbatch)):
        texts, golds = zip(*batch)
        try:
            nlp.update(texts, golds, optim, debug=True)
        except:
            fail_path = os.path.abspath("fail.json")
            log.error(f"Fail. Saved in {fail_path}")
            with open(fail_path, "w") as f:
                json.dump(batch, f, ensure_ascii=False)
            raise
        log.info(f"epoch {epoch} {j*cfg.nbatch}/{cfg.ndata}")
        if j % cfg.neval == cfg.neval - 1:
            scorer = eval_fn(cfg, nlp, val_data)
            log.info(f"ents_p: {scorer.ents_p}")
            log.info(f"ents_r: {scorer.ents_r}")
            log.info(f"ents_f: {scorer.ents_f}")


def train(cfg: Config, nlp, train_data, val_data, savedir: Path, eval_fn=None):
    if eval_fn is None:
        eval_fn = evaluate
    optim = nlp.resume_training(t_total=cfg.niter, enable_scheduler=cfg.scheduler)
    for i in range(cfg.niter):
        random.shuffle(train_data)
        train_epoch(cfg, nlp, optim, train_data, val_data, i, eval_fn)
        scorer = eval_fn(cfg, nlp, val_data)
        nlp.meta.update({"score": scorer.scores, "config": cfg.to_container()})
        nlp.to_disk(savedir / str(i))


def _main(cfg: Config):
    log.info(cfg.pretty())
    log.info("output dir: {}".format(os.getcwd()))
    train_data, val_data = create_data(cfg)
    nlp = create_nlp(cfg)
    if torch.cuda.is_available():
        log.info("CUDA enabled")
        nlp.to(torch.device("cuda"))
    savedir = Path.cwd() / "models"
    savedir.mkdir()
    train(cfg, nlp, train_data, val_data, savedir)


main = hydra.main(config_path="conf/train.yml")(_main)

if __name__ == "__main__":
    main()
