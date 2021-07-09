#!/usr/bin/env python3

import os
from typing import List, Optional
import fire
import subprocess


def build(dockerfile: str, tagname: str, python_version: str, workdir: str):
    cmd = [
        "docker",
        "build",
        "-f",
        dockerfile,
        "--build-arg",
        f"PYTHON_VERSION={python_version}",
        "--build-arg",
        f"WORKDIR={workdir}",
        "-t",
        tagname,
        ".",
    ]
    print("Build: ", " ".join(cmd))
    subprocess.run(cmd, check=True)


def test(tagname: str, cmd: List[str]):
    run_cmd = ["docker", "run"]
    run_cmd.append(tagname)
    run_cmd.extend(cmd)
    subprocess.run(run_cmd, check=True)


def main(
    python_version: str,
    package: Optional[str] = None,
    extra: Optional[str] = None,
    no_build: bool = False,
):

    dockerfile = "dockerfiles/Dockerfile"
    if extra:
        dockerfile += "." + extra
    tagname = f"camphr_{python_version}"
    if package:
        tagname += "_" + package
    if extra:
        tagname += "_" + extra

    if package:
        workdir = os.path.join("subpackages", package)
        test_cmd = ["../../test_local.sh", package]
    else:
        workdir = None
        test_cmd = ["./test_local.sh", "camphr"]

    # build
    if not no_build:
        build(dockerfile, tagname, python_version, workdir=workdir or ".")
    test(tagname, test_cmd)


if __name__ == "__main__":
    fire.Fire(main)
