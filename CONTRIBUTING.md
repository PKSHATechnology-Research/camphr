# Contributing


## Contributing to the code base

### Setup and Test

Camphr uses [poetry package manager](https://github.com/python-poetry/poetry), and [pre-commit](https://pre-commit.com/).

```bash
$ git clone https://github.com/PKSHATechnology-Research/camphr
$ poetry install
$ pre-commit install
$ poetry run pytest tests
```

### Test Udify and Elmo

Udify and Elmo are [extras dependencies](https://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-extras-optional-features-with-their-own-dependencies), and testing them requires model parameters.

Udify:

```bash
$ poetry install -E udify
$ poetry run python scripts/download_model.py en_udify
```

Elmo:

```bash
$ poetry install -E udify
$ poetry run python scripts/download_model.py en_elmo_medium
```

### Test MeCab, KNP (Japanese pipeline)

For testing MeCab or KNP, you need to install `mecab` or `knp` in your system respectively.
After that, put the following command:

```bash
$ poetry install -E mecab
# or
$ poetry install -E juman
```

# For maintainer

## Publish pypi package

Publishing a package to pypi is automatically done by specific git commit to this repository.
See `.github/workflows/fire_release.yml` for details.
