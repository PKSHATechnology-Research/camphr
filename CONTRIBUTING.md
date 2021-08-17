# Contributing

## Project Structure

Camphr consists of several subpackages in [./packages/](./packages) directory:

- camphr: Core Camphr package, including tokenizers and some basic interfaces (e.g. `DocProto`)
- camphr_transformers: Huggingface Transformers integration

See each package directory for details.

## Setup and Test

Camphr uses [poetry package manager](https://github.com/python-poetry/poetry).

```bash
$ git clone https://github.com/PKSHATechnology-Research/camphr

# test in container
# testing 'camphr' package in python3.8 environment with Dockerfile.base
$ python test_docker.py 3.8 camphr base
```

See each package directory for details.

# For maintainer

## Publish pypi package

Publishing packages to PYPI is done automatically by specific git tags.

1. `pip install pyversionup` (https://github.com/tamuhey/pyversionup)
2. `cd packages/$PKG_TO_PUBLISH`: move to the package directory you want to publish
3. `versionup $NEW_VERSION`: modify version strings
4. `git push --tags`: push tags to GitHub, which fires the GitHub Action

See `.github/workflows/main.yml` for details.

