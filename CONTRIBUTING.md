# Contributing


## Contributing to the code base

### Structure

Camphr consists of several subpackages in [./packages/](./packages) directory:

- camphr: Core Camphr package, including tokenizers and some basic interfaces (e.g. `DocProto`)
- camphr_transformers: Huggingface Transformers integration

See each package directory for details.

### Setup and Test

Camphr uses [poetry package manager](https://github.com/python-poetry/poetry).

```bash
$ git clone https://github.com/PKSHATechnology-Research/camphr
$ poetry install
$ poetry run pytest tests
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
