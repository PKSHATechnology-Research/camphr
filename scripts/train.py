import json
import logging
import operator
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import hydra
import numpy as np
import omegaconf
import srsly
import torch
from camphr.models import get_trf_name, trf_ner, trf_seq_classification
from camphr.ner_labels import LABELS
from camphr.ner_labels.utils import make_biluo_labels
from camphr.torch_utils import goldcat_to_label
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from spacy.language import Language
from spacy.scorer import Scorer
from spacy.util import minibatch
from typing_extensions import Literal

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
    labels: List[str]
    label: str
    scheduler: bool
    test_size: float
    lang: str
    pretrained: str
    neval: int
    task: Literal["textcat", "ner"]


def create_data(cfg: Config) -> Tuple[List, List]:
    data = load_data(cfg.data)
    if cfg.ndata != -1:
        data = random.sample(data, k=cfg.ndata)
    else:
        cfg.ndata = len(data)
    train, val = train_test_split(data, test_size=cfg.test_size)
    srsly.write_jsonl(Path.cwd() / f"train-data.jsonl", train)
    srsly.write_jsonl(Path.cwd() / f"val-data.jsonl", val)
    return train, val


def get_labels_from_file(cfg: Config) -> List[str]:
    if cfg.label_json:
        return srsly.read_json(os.path.expanduser(cfg.label_json))
    return []


def get_labels(cfg: Config) -> List[str]:
    labels = get_labels_from_file(cfg)
    if not labels:
        if cfg.task == "textcat":
            labels = cfg.labels
        if cfg.task == "ner":
            labels = make_biluo_labels(LABELS[cfg.label])
    if labels:
        return labels
    raise ValueError()


def create_nlp(cfg: Config) -> Language:
    labels = get_labels(cfg)
    nlp = None
    if cfg.task == "ner":
        nlp = trf_ner(
            lang=cfg.lang, pretrained=cfg.pretrained, labels=labels, freeze=cfg.freeze
        )
    elif cfg.task == "textcat":
        nlp = trf_seq_classification(
            lang=cfg.lang,
            pretrained=cfg.pretrained,
            labels=labels,
            label_weights=cfg.weights,
            freeze=cfg.freeze,
        )
    name = get_trf_name(cfg.pretrained)
    nlp.meta["name"] = name.value + "_" + (cfg.name or cfg.task)
    return nlp


def evaluate(cfg: Config, nlp: Language, val_data) -> Dict:
    try:
        scorer: Scorer = nlp.evaluate(val_data, batch_size=cfg.nbatch * 2)
    except:
        with open("fail.json", "w") as f:
            json.dump(val_data, f, ensure_ascii=False)
            raise
    return scorer.scores


def get_max_key(x):
    return max(x.items(), key=operator.itemgetter(1))[0]


def evaluate_textcat(cfg: Config, nlp: Language, val_data) -> Dict:
    # TODO: https://github.com/explosion/spaCy/pull/4664
    texts, golds = zip(*val_data)
    try:
        y = np.array(list(map(lambda x: goldcat_to_label(x["cats"]), golds)))
        docs = list(nlp.pipe(texts, batch_size=cfg.nbatch * 2))
        preds = np.array([get_max_key(doc.cats) for doc in docs])
    except:
        with open("fail.json", "w") as f:
            json.dump(val_data, f, ensure_ascii=False)
            raise
    return classification_report(y, preds, output_dict=True)


def get_eval_fn(cfg: Config):
    if cfg.task == "textcat":
        return evaluate_textcat
    return evaluate


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


def train(cfg: Config, nlp, train_data, val_data, savedir: Path):
    eval_fn = get_eval_fn(cfg)
    optim = nlp.resume_training(t_total=cfg.niter, enable_scheduler=cfg.scheduler)
    for i in range(cfg.niter):
        random.shuffle(train_data)
        train_epoch(cfg, nlp, optim, train_data, val_data, i, eval_fn)
        scores = eval_fn(cfg, nlp, val_data)
        nlp.meta.update({"score": scores, "config": cfg.to_container()})
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
