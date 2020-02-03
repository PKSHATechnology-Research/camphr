import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Union

import hydra
import srsly
from omegaconf import Config
from sklearn.model_selection import train_test_split

from camphr.utils import get_by_dotkey

GoldParsable = Dict[str, Any]
InputData = List[Tuple[str, GoldParsable]]

logger = logging.getLogger(__name__)


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
        logger.error(f"Error raised. The input data is saved in {str(fail_path)}")


def convert_fullpath_if_path(text: str) -> str:
    path = os.path.expanduser(text)
    path = hydra.utils.to_absolute_path(text)
    if os.path.exists(path):
        return path
    return text


def check_nonempty(cfg: Config, fields: Sequence[Union[str, Sequence[str]]]):
    errors = []
    for key in fields:
        if isinstance(key, str) and (not get_by_dotkey(cfg, key)):
            errors.append(f"{key} is required.")
        elif isinstance(key, list) and (not any(get_by_dotkey(cfg, k) for k in key)):
            errors.append(f"Any of {', '.join(key)} is required.")
    if errors:
        raise ValueError("\n".join(errors))
