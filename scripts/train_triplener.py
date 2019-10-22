import os
import logging
from pathlib import Path

import hydra
import torch

from train import Config, create_data, train
from bedoner.ner_labels import LABELS
from bedoner.ner_labels.utils import make_biluo_labels
from bedoner.models import get_trf_name, trf_ner, trf_ner_layer
from spacy.language import Language

log = logging.getLogger(__name__)


def get_top_label(label: str) -> str:
    return label.split("/")[0]


def get_second_top_label(label: str) -> str:
    if len(label) == 1:
        return label
    parts = label.split("/")
    if len(parts) == 1:
        return label
    pref = parts[0][:2]
    return pref + parts[1]


def create_nlp(cfg: Config) -> Language:
    labels = make_biluo_labels(LABELS[cfg.label])
    secondtop_labels = list({get_second_top_label(k) for k in labels})
    toplabels = list({k.split("/")[0] for k in labels})
    log.info(f"label1: {len(toplabels)}")
    log.info(f"label2: {len(secondtop_labels)}")
    log.info(f"label3: {len(labels)}")
    nlp = trf_ner(
        lang=cfg.lang,
        pretrained=cfg.pretrained,
        labels=toplabels,
        user_hooks={"convert_label": get_top_label},
    )
    ner = trf_ner_layer(
        lang=cfg.lang,
        pretrained=cfg.pretrained,
        vocab=nlp.vocab,
        labels=secondtop_labels,
        user_hooks={"convert_label": get_second_top_label},
    )
    ner.name = ner.name + "2"
    nlp.add_pipe(ner)

    ner3 = trf_ner_layer(
        lang=cfg.lang, pretrained=cfg.pretrained, vocab=nlp.vocab, labels=labels
    )
    ner3.name = ner3.name + "3"
    nlp.add_pipe(ner3)

    name = get_trf_name(cfg.pretrained)
    nlp.meta["name"] = name.value + "_" + cfg.label + "_triple"
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
    train(cfg, nlp, train_data, val_data, savedir)


main = hydra.main(config_path="conf/train.yml")(_main)

if __name__ == "__main__":
    main()
