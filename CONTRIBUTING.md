# Contributing


## Contributing to the code base

### Setup and Test

Camphr uses [poetry package manager](https://github.com/python-poetry/poetry), and [pre-commit](https://pre-commit.com/).

```bash
$ git clone https://github.com/PKSHATechnology-Research/camphr
$ poetry install
$ pre-commit intall
$ poetry run pytest tests
```

### Test Udify and Elmo

Udify and Elmo are [extras dependencies](https://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-extras-optional-features-with-their-own-dependencies), and testing them requires model parameters.

```bash
$ poetry install -E allennlp udify
$ make download
```

### Test MeCab, KNP (Japanese pipeline)

MeCabやKNPをするには，システムにmecabやknpがインストールされている必要があります．
また，以下のようにして`pyknp`と`mecab-python3`をインストールする必要があります.

```bash
$ poetry install -E mecab
# or
$ poetry install -E juman
```
