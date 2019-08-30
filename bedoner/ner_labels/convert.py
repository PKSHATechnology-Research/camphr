"""Extract keys in yaml and generate python script"""
from typing import Dict, Optional, List, IO
from pathlib import Path
import yaml

__dir__ = Path(__file__).parent


def extract_keys(d: Dict[str, Optional[Dict]]) -> List[str]:
    res = []
    for k, v in d.items():
        res.append(k)
        if isinstance(v, dict):
            res += extract_keys(v)
    return res


def generate_py(f: IO, keys: List[str]):
    for k in keys:
        f.write(f"{k} = '{k}'\n")


PREFIX = "labels_"
if __name__ == "__main__":
    for y in __dir__.glob("*.yml"):
        with y.open() as f, (__dir__ / (PREFIX + y.stem + ".py")).open("w") as fw:
            d = yaml.safe_load(f)
            keys = sorted(extract_keys(d))
            generate_py(fw, keys)
