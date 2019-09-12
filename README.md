# NER統合 ![CircleCI](https://circleci.com/gh/PKSHATechnology/bedore-ner.svg?style=svg)

# Installation

```bash
$ pipenv install .
$ pipenv install --dev -e . # for developer
```

## requrements

- mecab, juman(pp), knpが必要です．

## packaged piplines

- bedore-aws-solution-randdの`s3://bedoner/releases`からtar.gzをダウンロードしてください
- `pip install foo.tar.gz`でOKです．
	- 必要な辞書，パラメータ等全てセットアップされます．

### 例

- ルールベースNER

```bash
$ aws s3 cp s3://bedoner/releases/mecab_entity_ruler-0.0.0.tar.gz .
$ pip install mecab_entity_ruler-0.0.0.tar.gz
```
```python
>> import spacy
>> nlp = spacy.load("mecab_entity_ruler")
>> nlp("2019年11月8日に高松隆と東京タワーに行った").ents
(2019年11月8日, 高松隆)
```

- BERT NER

```bash
$ aws s3 cp s3://bedoner/releases/juman_bert_ner-0.0.0.tar.gz .
$ pip install juman_bert_ner-0.0.0.tar.gz .
```
```python
>> import spacy
>> nlp = spacy.load("juman_bert_ner")
>> nlp("EXILEのATSUSHIと中島美嘉が14日ニューヨーク入り").ents
("ＥＸＩＬＥ", "ＡＴＳＵＳＨＩ", "中島美嘉", "１４日", "ニューヨーク")
```

- KNP NERとルールベースの組み合わせ

```bash
$ aws s3 cp s3://bedoner/releases/knp_entity_extractor-0.0.0.tar.gz .
$ pip install knp_entity_extractor-0.0.0.tar.gz
```
```python
>>> import spacy
>>> nlp = spacy.load("knp_entity_extractor")
>>> nlp("今日はPKSHAを訪問した").ents
(今日,)

>>> ruler = nlp.create_pipe("entity_ruler")
>>> ruler.add_patterns([{"label": "ORG", "pattern": [{"TEXT": "PKSHA"}]}])
>>> nlp.add_pipe(ruler)
>>> nlp("今日はPKSHAを訪問した").ents
(今日, PKSHA)
```

# Development

- 開発にあたり，spacyの仕組みについて公式ドキュメント([Architecture · spaCy API Documentation](https://spacy.io/api))を一読してください

## setup

1. clone
2. `$ pipenv install --dev -e .`
3. `$ make download`
4. `$ pipenv run pytest`

## 構成

- `bedoner`: package source
- `tests`: test files
- `scripts`: 細々としたスクリプト. packagingに必要なものなど

## packaging

- [Saving and Loading · spaCy Usage Documentation](https://spacy.io/usage/saving-loading)
- `cd scripts && pipenv run papermill packaging.ipynb log.ipynb`


## Refs.

- attlassian 
	- https://pkshatech.atlassian.net/wiki/spaces/CHAT/pages/35061918/NER+2019+8
- NERのタグ
	- https://gist.github.com/kzinmr/14c224efc43b7e21ff95fa9c54f829f1
