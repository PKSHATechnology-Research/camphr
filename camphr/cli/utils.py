import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import hydra
import srsly
from camphr.models import correct_model_config
from camphr.utils import create_dict_from_dotkey, get_by_dotkey
from omegaconf import Config, OmegaConf
from sklearn.model_selection import train_test_split
from spacy.language import Language
from spacy.scorer import Scorer

GoldParsable = Dict[str, Any]
InputData = List[Tuple[str, GoldParsable]]

log = logging.getLogger(__name__)


def create_data(cfg: Config) -> Tuple[InputData, InputData]:
    data = list(srsly.read_jsonl(Path(cfg.path).expanduser()))
    if cfg.ndata > 0:
        data = random.sample(data, k=cfg.ndata)
    else:
        cfg.ndata = len(data)
    train, val = train_test_split(data, test_size=cfg.val_size)
    srsly.write_jsonl(Path.cwd() / f"train-data.jsonl", train)
    srsly.write_jsonl(Path.cwd() / f"val-data.jsonl", val)
    return train, val


def report_fail(json_serializable_data: Any) -> None:
    fail_path = Path("fail.json").absolute()
    with fail_path.open("w") as f:
        json.dump(json_serializable_data, f, ensure_ascii=False)
        log.error(f"Error raised. The input data is saved in {str(fail_path)}")


def evaluate(cfg: Config, nlp: Language, val_data: InputData) -> Dict:
    try:
        scorer: Scorer = nlp.evaluate(val_data, batch_size=cfg.nbatch * 2)
    except Exception:
        report_fail(val_data)
        raise
    return scorer.scores


def convert_fullpath_if_path(text: str) -> str:
    path = hydra.utils.to_absolute_path(text)
    if os.path.isfile(path):
        return path
    return text


def check_nonempty(cfg: Config, fields: List[Union[str, List[str]]]):
    errors = []
    for key in fields:
        if isinstance(key, str):
            if not get_by_dotkey(cfg, key):
                errors.append(f"{key} is required.")
        elif isinstance(key, list):
            if not any(get_by_dotkey(cfg, k) for k in key):
                errors.append(f"Any of {', '.join(key)} is required.")
    if errors:
        raise ValueError("\n".join(errors))


def validate(cfg: Config):
    mustfields = [
        "train.data.path",
        [
            "model.ner_label",
            "model.textcat_label",
            "model.pipeline.transformers_ner.labels",
            "model.pipeline.transformers_seq_classification.labels",
        ],
        "model.lang.name",
    ]
    check_nonempty(cfg, mustfields)

    pathkey = ["model.ner_label", "model.textcat_label", "train.data.path"]
    for key in pathkey:
        path = get_by_dotkey(cfg, key)
        if path:
            path = convert_fullpath_if_path(path)
            cfg = OmegaConf.merge(
                cfg, OmegaConf.create(create_dict_from_dotkey(key, path))
            )

    cfg.model = correct_model_config(cfg.model)
    return cfg
