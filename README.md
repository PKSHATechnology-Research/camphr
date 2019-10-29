# NER on spaCy [![CircleCI](https://circleci.com/gh/PKSHATechnology/bedore-ner.svg?style=svg&circle-token=d27152116259f09d7e229ee7d5ad5f095989fc7d)](https://circleci.com/gh/PKSHATechnology/bedore-ner)

# Installation

```bash
$ pipenv install .
$ pipenv install --dev # for developer
```

## Model installation

1. [リリースページ](https://github.com/PKSHATechnology/bedore-ner/releases)からtar.gzをダウンロードします
2. `pip installl -U ...tar.gz` でOK!

Username, Passwordが聞かれるかもしれません．GitHubのユーザ名と，トークンを入れてください．
これは裏でbedore-nerレポジトリをcloneしているためです．

### ブラウザ以外でダウンロードしたい場合

GitHub APIを使う必要があります．

`pip install git+https://github.com/tamuhey/github_asset`としたのち，`github-asset get file_to_download`でダウンロードできます．

## Requirements

- 使う機能によっては， mecab, juman(pp), knpが必要です．

# Usage

[docs/pipelines](./docs/usage/README.md) をみてください．

# Development

- 開発にあたり，spacyの仕組みについて公式ドキュメント([Architecture · spaCy API Documentation](https://spacy.io/api))を一読することをお勧めします．
- 開発者向けドキュメントを [docs/development](./docs/development)に集めています．

## setup

1. clone
2. `$ pipenv install --dev`
3. `$ make download`
4. `$ pipenv run pytest`

## 構成

- `bedoner`: package source
- `tests`: test files
- `scripts`: 細々としたスクリプト. packagingに必要なものなど

## packaging

- [Saving and Loading · spaCy Usage Documentation](https://spacy.io/usage/saving-loading)


## Refs.

- attlassian 
	- https://pkshatech.atlassian.net/wiki/spaces/CHAT/pages/35061918/NER+2019+8
- NERのタグ
	- https://gist.github.com/kzinmr/14c224efc43b7e21ff95fa9c54f829f1
