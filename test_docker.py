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
    install_cmd: str,
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

    cmd_str = " ".join(cmd)
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise ValueError(f"{cmd_str} failed.") from e
    print("Build Finished: ", cmd_str)


def test(tagname: str, cmd: List[str]):
    run_cmd = ["docker", "run"]
    run_cmd.append(tagname)
    run_cmd.extend(cmd)
    subprocess.run(run_cmd, check=True)


def main(
    python_version: str,
    package: str,
    dockerfile_ext: str,
    install_cmd: str = "poetry install",
    no_build: bool = False,
):
    """Running test with docker"""
    dockerfile = f"dockerfiles/Dockerfile.{dockerfile_ext}"
    tagname = f"camphr_{python_version}_{package}_{dockerfile_ext}"
    test_cmd = ["/root/test_local.sh", package]
    workdir = os.path.join("packages", package)

    if not os.path.exists(workdir):
        raise ValueError(f"{workdir} not exist")

    # build
    if not no_build:
        build(
            dockerfile,
            tagname,
            python_version,
            workdir=workdir,
            install_cmd=install_cmd,
        )
    else:
        print("Build skipped")
    test(tagname, test_cmd)


if __name__ == "__main__":
    fire.Fire(main)
