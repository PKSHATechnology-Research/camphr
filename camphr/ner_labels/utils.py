"""Module utils defines utility for ner labels.

This module can be called directly to generete "labels_**.py"

Examples:
    $ python utils.py
"""
import yaml  # type: ignore
import subprocess
from pathlib import Path
from typing import IO, Dict, Iterable, List, Optional, Union
import json

from typing_extensions import Literal


__dir__ = Path(__file__).parent

NER_LABEL_TYPE = Literal["BIO", "BILUO"]


def make_ner_labels(
    entity_types: Iterable[str], type_: NER_LABEL_TYPE = "BIO"
) -> List[str]:
    """Make BILUO or BIO style ner label from entity type list.

    Examples:
        >>> make_ner_labels(["PERSON"])
        ["-", "O", "U-PERSON", "B-PERSON", ...]
    """
    labels = ["-", "O"]
    prefixes = [p for p in type_ if p != "O"]
    for label in entity_types:
        for pref in prefixes:
            labels.append(pref + "-" + label)
    return list(dict.fromkeys(labels))  # unique while keep ordering


def _get_labels(labels_or_path: Union[List[str], Path, str]) -> List[str]:
    if isinstance(labels_or_path, (str, Path)):
        path = Path(labels_or_path)
        if path.suffix == ".json":
            return json.loads(path.read_text())
        elif path.suffix in {".yml", "yaml"}:
            return yaml.safe_load(path.read_bytes())
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")
    return labels_or_path


def get_ner_labels(labels: Union[List[str], Path], type_: NER_LABEL_TYPE = "BIO"):
    labels = _get_labels(labels)
    if all(label[:2] in {"-", "O", "I-", "B-", "L-", "U-"} for label in labels):
        return labels
    return make_ner_labels(labels, type_)


def _yml_to_py(yml_path: Union[str, Path], py_path: Union[str, Path]):
    """Convert yml file that defines ner labels to python script."""
    yml_path = Path(yml_path)
    py_path = Path(py_path)
    with yml_path.open() as f, py_path.open("w") as fw:
        d = yaml.safe_load(f)
        keys = sorted(_extract_keys(d))
        _generate_py(fw, keys)


_LABEL_DICT = Dict[str, Optional["_LABEL_DICT"]]


def _extract_keys(d: _LABEL_DICT, prefix: str = "") -> List[str]:
    """Nested key are described with slash: e.g. Animal.Bird -> Animal/Bird"""
    res: List[str] = []
    for k, v in d.items():
        res.append((prefix + k).upper())
        if isinstance(v, dict):
            res += _extract_keys(v, prefix + k + "/")
    return res


def _crean_slash(key: str) -> str:
    return key.split("/")[-1]


def _generate_py(f: IO[str], keys: List[str]):
    for k in keys:
        f.write(f"{_crean_slash(k)} = '{k}'\n")
    f.write(f'\nALL_LABELS = [{" ,".join(map(_crean_slash, keys))}]\n')


if __name__ == "__main__":
    PREFIX = "labels_"
    for y in __dir__.glob("*.yml"):
        p = __dir__ / (PREFIX + y.stem + ".py")
        _yml_to_py(y, p)
    subprocess.run(["black", "."])
