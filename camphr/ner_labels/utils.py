"""Module utils defines utility for ner labels.

This module can be called directly to generete "labels_**.py" from yml file.

Examples:
    $ python utils.py
"""
import subprocess
from pathlib import Path
from typing import IO, Dict, Iterable, List, Optional, Union

import yaml
from typing_extensions import Literal

from camphr.types import Pathlike
from camphr.utils import get_labels

__dir__ = Path(__file__).parent


def make_ner_labels(
    entity_types: Iterable[str], type_: Literal["BIO", "BILUO"] = "BIO"
) -> List[str]:
    """Make BILUO or BIO style ner label from entity type list.

    Examples:
        >>> make_ner_labels(["PERSON"])
        ["-", "O", "U-PERSON", "B-PERSON", ...]
    """
    labels = ["-", "O"]
    prefixes = [p for p in type_ if p != "O"]
    for l in entity_types:
        for pref in prefixes:
            labels.append(pref + "-" + l)
    return list(dict.fromkeys(labels))  # unique while keep ordering


def get_ner_labels(labels: Union[List[str], Pathlike], type_="BIO"):
    labels = get_labels(labels)
    if all(l[:2] in {"-", "O", "I-", "B-", "L-", "U-"} for l in labels):
        return labels
    return make_ner_labels(labels, type_)


def yml_to_py(yml_path: Union[str, Path], py_path: Union[str, Path]):
    """Convert yml file that defines ner labels to python script."""
    yml_path = Path(yml_path)
    py_path = Path(py_path)
    with yml_path.open() as f, py_path.open("w") as fw:
        d = yaml.safe_load(f)
        keys = sorted(extract_keys(d))
        generate_py(fw, keys)


def extract_keys(d: Dict[str, Optional[Dict]], prefix: str = "") -> List[str]:
    """Nested key are described with slash: e.g. Animal.Bird -> Animal/Bird"""
    res = []
    for k, v in d.items():
        res.append((prefix + k).upper())
        if isinstance(v, dict):
            res += extract_keys(v, prefix + k + "/")
    return res


def _crean_slash(key: str) -> str:
    return key.split("/")[-1]


def generate_py(f: IO, keys: List[str]):
    for k in keys:
        f.write(f"{_crean_slash(k)} = '{k}'\n")
    f.write(f'\nALL_LABELS = [{" ,".join(map(_crean_slash, keys))}]\n')


PREFIX = "labels_"
if __name__ == "__main__":
    for y in __dir__.glob("*.yml"):
        p = __dir__ / (PREFIX + y.stem + ".py")
        yml_to_py(y, p)
    subprocess.run(["black", "."])
