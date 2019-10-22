import os
import logging
from pathlib import Path
import json

import hydra
import torch

from train import Config, create_data, train
from bedoner.ner_labels import LABELS
from bedoner.ner_labels.utils import make_biluo_labels
from bedoner.models import get_trf_name, trf_ner, trf_ner_layer
from spacy.language import Language
from spacy.scorer import Scorer

log = logging.getLogger(__name__)


def get_top_label(label: str) -> str:
    return label.split("/")[0]


def second_label(label: str) -> str:
    if len(label) == 1:
        return label
    items = label.split("/")
    if len(items) == 1:
        return label


def evaluate(cfg: Config, nlp: Language, val_data) -> Scorer:
    try:
        with nlp.disable_pipes("xlnet_ner2"):
            scorer: Scorer = nlp.evaluate(val_data, batch_size=cfg.nbatch * 2)
    except:
        with open("fail.json", "w") as f:
            json.dump(val_data, f, ensure_ascii=False)
            raise
    return scorer


def create_nlp(cfg: Config) -> Language:
    labels = make_biluo_labels(LABELS[cfg.label])
    toplabels = list({k.split("/")[0] for k in labels})
    nlp = trf_ner(
        lang=cfg.lang,
        pretrained=cfg.pretrained,
        labels=toplabels,
        user_hooks={"convert_label": get_top_label},
    )
    ner = trf_ner_layer(
        lang=cfg.lang, pretrained=cfg.pretrained, vocab=nlp.vocab, labels=labels
    )
    ner.name = ner.name + "_sekine"
    nlp.add_pipe(ner)
    name = get_trf_name(cfg.pretrained)
    nlp.meta["name"] = name.value + "_" + cfg.label
    return nlp


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
    train(cfg, nlp, train_data, val_data, savedir, eval_fn=evaluate)


main = hydra.main(config_path="conf/train.yml")(_main)

if __name__ == "__main__":
    main()
