#!/usr/bin/env python
# coding: utf8
from __future__ import unicode_literals

import io
import json
from os import path, walk
from shutil import copy
from setuptools import setup
import subprocess
import sys


def load_meta(fp):
    with io.open(fp, encoding="utf8") as f:
        return json.load(f)


def list_files(data_dir):
    output = []
    for root, _, filenames in walk(data_dir):
        for filename in filenames:
            if not filename.startswith("."):
                output.append(path.join(root, filename))
    output = [path.relpath(p, path.dirname(data_dir)) for p in output]
    output.append("meta.json")
    return output


def list_requirements(meta):
    parent_package = meta.get("parent_package", "spacy")
    requirements = [parent_package + meta["spacy_version"]]
    if "setup_requires" in meta:
        requirements += meta["setup_requires"]
    if "requirements" in meta:
        requirements += meta["requirements"]
    return requirements


def setup_package():
    if subprocess.call(["make", "create"], cwd="jinmei") != 0:
        sys.exit(-1)

    root = path.abspath(path.dirname(__file__))
    meta_path = path.join(root, "meta.json")
    meta = load_meta(meta_path)
    model_name = str(meta["lang"] + "_" + meta["name"])
    model_dir = path.join(model_name, model_name + "-" + meta["version"])

    copy(meta_path, path.join(model_name))
    copy(meta_path, model_dir)
    copy(path.join(root, "jinmei", "user.dic"), path.join(model_dir, "tokenizer"))

    setup(
        name=model_name,
        description=meta["description"],
        author=meta["author"],
        author_email=meta["email"],
        url=meta["url"],
        version=meta["version"],
        license=meta["license"],
        packages=[model_name],
        package_data={model_name: list_files(model_dir), "jinmei": ["*"]},
        install_requires=list_requirements(meta),
        zip_safe=False,
    )


if __name__ == "__main__":
    setup_package()
