import subprocess
import sys
from pathlib import Path

pkgd = Path("pkgs")
fmodels = sys.argv[1]
models = list(open(fmodels).read().split())
for model in models:
    print(model)
    path = pkgd / model / f"{model.split('-')[0]}" / model
    if path.exists():
        subprocess.run(["python", "scripts/versionup_package.py", str(path), "v0.4"])
