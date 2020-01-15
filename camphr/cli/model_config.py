from pathlib import Path

import fire
import yaml

MODEL_CONFIG_PATH = Path(__file__).parent / "../model_config"


def main(name: str = "", ls: bool = False):
    if ls:
        for fpath in MODEL_CONFIG_PATH.glob("**/*.yml"):
            with fpath.open() as f:
                cfg = yaml.safe_load(f)
                print(cfg["name"])
        return
    fpath = MODEL_CONFIG_PATH / f"{name}.yml"
    print(fpath.read_text())


if __name__ == "__main__":
    fire.Fire(main)
