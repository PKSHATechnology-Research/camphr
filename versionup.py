import sys
import configparser
from pathlib import Path
from typing import Iterable
import subprocess


def versionup(p: Path, old, new):
    with p.open() as f:
        text = f.read()
    with p.open("w") as f:
        f.write(text.replace(old, new))


def rewrite_version(filenames: Iterable[str], old, new):
    for fname in filenames:
        if fname:
            p = Path(fname)
            versionup(p, old, new)
            print(f"rewrite version: {str(p)}")


def call(cmd):
    proc = subprocess.run(cmd, capture_output=True)
    print(proc.stdout.decode())
    print(proc.stderr.decode())


def commit(old_version, new_version):
    cmd = [
        "git",
        "commit",
        "--allow-empty",
        "-m",
        f'"versionup: {old_version} -> {new_version}"',
    ]
    call(cmd)


def tag(tagname):
    cmd = ["git", "tag", tagname]
    call(cmd)


def main():
    config = configparser.ConfigParser()
    config.read("setup.cfg")
    old_version = config["metadata"]["version"]
    new_version = sys.argv[1]
    config["metadata"]["version"] = new_version
    with open("setup.cfg", "w") as f:
        config.write(f)

    if "versioning" in config:
        vcfg = config["versioning"]
        rewrite_version(vcfg["files"].split("\n"), old_version, new_version)

        if vcfg.get("commit") == "True":
            commit(old_version, new_version)
        if vcfg.get("tag") == "True":
            tag(new_version)


if __name__ == "__main__":
    main()
