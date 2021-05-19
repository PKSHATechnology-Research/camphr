import json
import logging
import os
from pathlib import Path
import random
from typing import Any, Dict, List, Sequence, Tuple, TypeVar, Union, cast

from camphr_core.utils import dump_jsonl, get_by_dotkey
import hydra
from omegaconf import Config
from sklearn.model_selection import train_test_split

GoldParsable = Dict[str, Any]
InputData = List[Tuple[str, GoldParsable]]

logger = logging.getLogger(__name__)


def read_jsonl(path: Path) -> List[str]:
    ret = []
    with path.open("r") as f:
        for line in f:
            ret.append(json.loads(line))
    return ret


def create_data(cfg: Config) -> Tuple[InputData, InputData]:
    data = read_jsonl(Path(cfg.path).expanduser())
    if cfg.ndata > 0:
        data = random.sample(data, k=cfg.ndata)
    else:
        cfg.ndata = len(data)
    train, val = train_test_split(data, test_size=cfg.val_size)

    with (Path().cwd() / "train-data.jsonl").open("w") as f:
        dump_jsonl(f, train)
    with (Path.cwd() / "val-data.jsonl").open("w") as f:
        dump_jsonl(f, val)
    return train, val


def report_fail(json_serializable_data: Any) -> None:
    fail_path = Path("fail.json").absolute()
    with fail_path.open("w") as f:
        json.dump(json_serializable_data, f, ensure_ascii=False)
        logger.error(f"Error raised. The input data is saved in {str(fail_path)}")


def convert_fullpath_if_path(text: str) -> str:
    if not isinstance(text, str):
        raise ValueError(f"Expected str, got {type(text)}, value {text}")
    path = os.path.expanduser(text)
    try:
        path = cast(str, hydra.utils.to_absolute_path(path))
    except AttributeError:
        # Not in hydra runtime
        pass
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


T0 = TypeVar("T0")
T1 = TypeVar("T1")


def unzip2(x: List[Tuple[T0, T1]]) -> Tuple[Tuple[T0], Tuple[T1]]:
    # trick to convince typechecker
    a, b = zip(*x)
    return a, b  # type: ignore
