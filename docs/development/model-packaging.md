# How to package models

## BERT NER

0. トレーニングしたモデルを`nlp.to_disk`を使って保存します
1. `scripts/packaging.py`を使ってpackage化します
2. `scripts/test-packaging.sh`を使ってテストします
3. [github-asset](https://github.com/tamuhey/github_asset)等を使って，release pageにtarballをアップロードします