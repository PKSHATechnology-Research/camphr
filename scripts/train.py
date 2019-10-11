import hydra
import torch
from typing import List, Dict
import json
from spacy.scorer import Scorer
import random
from pathlib import Path

from spacy.util import minibatch
import logging
from sklearn.model_selection import train_test_split
import omegaconf

from bedoner.models import *
from bedoner.ner_labels.labels_irex import ALL_LABELS as irex_labels
from bedoner.ner_labels.labels_ene import ALL_LABELS as ene_labels
from bedoner.ner_labels.utils import make_biluo_labels, make_bio_labels

log = logging.getLogger(__name__)


def get_labels(name: str) -> List[str]:
    if name == "irex":
        return irex_labels
    elif name == "ene":
        return ene_labels
    raise ValueError(f"Unknown label type: {name}")


def load_data(name: str) -> List[Dict]:
    name = os.path.expanduser(name)
    data = []
    with open(name) as f:
        for line in f:
            data.append(json.loads(line))
    return data


class Config(omegaconf.Config):
    data: str
    ndata: int
    niter: int
    nbatch: int
    label: str
    scheduler: bool
    test_size: float
    lang: str
    name: str


@hydra.main(config_path="conf/train.yml")
def main(cfg: Config):
    log.info(cfg.pretty())
    outputd = os.getcwd()
    log.info("output dir: {}".format(outputd))
    data = load_data(cfg.data)
    if cfg.ndata != -1:
        data = random.sample(data, k=cfg.ndata)
    else:
        cfg.ndata = len(data)
    train_data, val_data = train_test_split(data, test_size=cfg.test_size)

    labels = get_labels(cfg.label)
    nlp = bert_ner(lang=cfg.lang, labels=make_biluo_labels(labels))
    nlp.meta["name"] = cfg.name
    if torch.cuda.is_available():
        log.info("CUDA enabled")
        nlp.to(torch.device("cuda"))

    optim = nlp.resume_training(t_total=cfg.niter, enable_scheduler=cfg.scheduler)
    modelsdir = Path.cwd() / "models"
    modelsdir.mkdir()

    for i in range(cfg.niter):
        random.shuffle(train_data)
        epoch_loss = 0
        for j, batch in enumerate(minibatch(train_data, size=cfg.nbatch)):
            texts, golds = zip(*batch)
            docs = [nlp.make_doc(text) for text in texts]
            try:
                nlp.update(docs, golds, optim)
            except:
                with open("fail.json", "w") as f:
                    json.dump(batch, f)
                raise
            loss = sum(doc._.loss.detach().item() for doc in docs)
            epoch_loss += loss
            log.info(f"{j*cfg.nbatch}/{cfg.ndata} loss: {loss}")
            if j % 10 == 9:
                scorer: Scorer = nlp.evaluate(val_data)
                log.info(f"p: {scorer.ents_p}")
                log.info(f"r: {scorer.ents_r}")
                log.info(f"f: {scorer.ents_f}")
        log.info(f"epoch {i} loss: {epoch_loss}")
        scorer: Scorer = nlp.evaluate(val_data)
        nlp.meta = {"score": scorer.scores, "config": cfg.to_container()}
        nlp.to_disk(modelsdir / str(i))


if __name__ == "__main__":
    main()
