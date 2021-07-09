#!/usr/bin/env python3

import os
from typing import List, Optional
import fire
import subprocess


def build(dockerfile: str, tagname: str, python_version: str):
    cmd = [
        "docker",
        "build",
        "-f",
        dockerfile,
        "--build-arg",
        f"PYTHON_VERSION={python_version}",
        "-t",
        tagname,
        ".",
    ]
    subprocess.run(cmd, check=True)


def test(tagname: str, cmd: List[str], workdir: Optional[str] = None):
    run_cmd = ["docker", "run"]
    if workdir:
        run_cmd.extend(["-w", workdir])
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

    # build
    if not no_build:
        build(dockerfile, tagname, python_version)

    # test
    if package:
        workdir = os.path.join("subpackages", package)
        cmd = ["../../test_local.sh", package]
    else:
        workdir = None
        cmd = ["./test_local.sh", "camphr"]
    test(tagname, cmd, workdir)


if __name__ == "__main__":
    fire.Fire(main)
