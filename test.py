#!/usr/bin/env python3

import os
from typing import List, Optional
import fire
import subprocess


def build(
    dockerfile: str,
    tagname: str,
    python_version: str,
    workdir: str,
    install_cmd: str = "poetry install",
):
    cmd = [
        "docker",
        "build",
        "-f",
        dockerfile,
        "--build-arg",
        f"PYTHON_VERSION={python_version}",
        "--build-arg",
        f"WORKDIR={workdir}",
        "--build-arg",
        f"INSTALL_CMD={install_cmd}",
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
    dockerfile_ext: Optional[str] = None,
    install_cmd: str = "poetry install",
    no_build: bool = False,
):
    dockerfile = "dockerfiles/Dockerfile"
    if dockerfile_ext:
        dockerfile += "." + dockerfile_ext
    tagname = f"camphr_{python_version}"
    if package:
        tagname += "_" + package
    if dockerfile_ext:
        tagname += "_" + dockerfile_ext

    test_cmd = ["/root/test_local.sh"]
    if package:
        workdir = os.path.join("subpackages", package)
        test_cmd.append(package)
    else:
        workdir = None
        test_cmd.append("camphr")

    # build
    if not no_build:
        build(
            dockerfile,
            tagname,
            python_version,
            workdir=workdir or ".",
            install_cmd=install_cmd,
        )
    else:
        print("Build skipped")
    test(tagname, test_cmd)


if __name__ == "__main__":
    fire.Fire(main)
