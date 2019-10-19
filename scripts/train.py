import json
import logging
import random
from pathlib import Path
from typing import Dict, List

import hydra
import omegaconf
import torch
from sklearn.model_selection import train_test_split
from spacy.scorer import Scorer
from spacy.util import minibatch

from bedoner.models import *
from bedoner.ner_labels.labels_ene import ALL_LABELS as ene_labels
from bedoner.ner_labels.labels_irex import ALL_LABELS as irex_labels
from bedoner.ner_labels.utils import make_biluo_labels

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
    from_pretrained: str
    neval: int


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
    nlp = bert_ner(
        lang=cfg.lang, pretrained=cfg.from_pretrained, labels=make_biluo_labels(labels)
    )
    nlp.meta["name"] = cfg.name + "_" + cfg.label
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
                    json.dump(batch, f, ensure_ascii=False)
                raise
            loss = sum(doc._.loss.detach().item() for doc in docs)
            epoch_loss += loss
            log.info(f"{j*cfg.nbatch}/{cfg.ndata} loss: {loss}")
            if j % cfg.neval == cfg.neval-1:
                try:
                    scorer: Scorer = nlp.evaluate(val_data)
                except:
                    with open("fail.json", "w") as f:
                        json.dump(val_data, f, ensure_ascii=False)
                        raise
                log.info(f"p: {scorer.ents_p}")
                log.info(f"r: {scorer.ents_r}")
                log.info(f"f: {scorer.ents_f}")
        log.info(f"epoch {i} loss: {epoch_loss}")
        try:
            scorer: Scorer = nlp.evaluate(val_data)
        except:
            with open("fail.json", "w") as f:
                json.dump(val_data, f, ensure_ascii=False)
            raise
        nlp.meta.update({"score": scorer.scores, "config": cfg.to_container()})
        nlp.to_disk(modelsdir / str(i))


if __name__ == "__main__":
    main()
