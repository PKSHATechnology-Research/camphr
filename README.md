# <img src="./img/logoc.png" width="20"> Camphr - Transformers on spaCy [![CircleCI](https://circleci.com/gh/PKSHATechnology/bedore-ner.svg?style=svg&circle-token=d27152116259f09d7e229ee7d5ad5f095989fc7d)](https://circleci.com/gh/PKSHATechnology/bedore-ner)

# Installation

```bash
$ pipenv install
$ pipenv install --dev # for developer
```

## Model installation

1. [リリースページ](https://github.com/PKSHATechnology/bedore-ner/releases)からtar.gzをダウンロードします
2. `pip installl -U ...tar.gz` でOK!

Username, Passwordが聞かれるかもしれません．GitHubのユーザ名と，トークンを入れてください．
これは裏でbedore-nerレポジトリをcloneしているためです．

### ブラウザ以外でダウンロードしたい場合

GitHub APIを使う必要があります．

```
$ pip install git+https://github.com/tamuhey/github_asset
```

としたのち，

```
github-asset get file_to_download
```

でダウンロードできます．(GitHubトークンを聞かれるので，入力してください)

## Requirements

- 使う機能によっては， mecab, juman(pp), knpが必要です．

# Usage

[docs/pipelines](./docs/usage/README.md) をみてください．

# Development

see [CONTRIBUTING.md](./CONTRIBUTING.md)