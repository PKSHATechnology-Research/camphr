import json
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

MAP_URL = "https://raw.githubusercontent.com/PKSHATechnology-Research/camphr_models/master/info.json"


def get_modelmap() -> Dict[str, Any]:
    with urllib.request.urlopen(MAP_URL) as f:
        return json.loads(f.read().decode())


def download(url: str, filename: Path, bufsize=10000):
    with urllib.request.urlopen(url) as f, filename.open("wb") as fw:
        shutil.copyfileobj(f, fw)


def download_model(name: str, version: Optional[str], directory: Optional[Path] = None):
    modelmap = get_modelmap()
    models = modelmap[name]
    model: Optional[Dict[str, Any]] = None
    if version:
        for e in models:
            if e["version"] == version:
                model = e
    else:
        model = models[0]
    assert model, (name, version)

    subprocess.call([sys.executable, "-m", "pip", "install", model["download_url"]])


def main():
    model_name = sys.argv[1]
    version = None
    if len(sys.argv) > 2:
        version = sys.argv[2]
    download_model(model_name, version)


if __name__ == "__main__":
    main()
