# NER統合 [![CircleCI](https://circleci.com/gh/PKSHATechnology/bedore-ner.svg?style=svg&circle-token=d27152116259f09d7e229ee7d5ad5f095989fc7d)](https://circleci.com/gh/PKSHATechnology/bedore-ner)

# Installation

```bash
$ pipenv install .
$ pipenv install --dev # for developer
```

## requrements

- mecab, juman(pp), knpが必要です．

# Examples

##  Date

```python
>>> from bedoner.models import date_ruler
>>> nlp = date_ruler()
>>> text = "2019年11月8日に高松隆と東京タワーに行った"
>>> nlp(text).ents
(2019年11月8日,)
```

## Person

```python
>>> from bedoner.models import person_ruler
>>> nlp = person_ruler()
>>> text = "2019年11月8日に高松隆と東京タワーに行った"
>>> nlp(text).ents
(高松隆,)
```

## KNP

```python
>>> from bedoner.models import knp_ner
>>> nlp = knp_ner()
>>> text = "2019年11月8日に高松隆と東京タワーに行った"
>>> nlp(text).ents
(2019年11月8日, 高松隆, 東京タワー)
```

## BERT

[リリースページ](https://spacy.io/api/annotation#json-input)からトレーニング済みモデルをダウンロードして，以下のようにpipでインストールしてください.  
パラメータ等全て入っています．

```bash
$ pip install juman-bert-irex.VERSION.tar.gz
```

```python
>>> import spacy
>>> nlp=spacy.load("juman_bert_irex")
>>> nlp("フランスのシラク元大統領が26日、死去した。86歳だった。フランスメディアが伝えた。").ents
(フランス, シラク, 元大統領, ２６日, フランス)
```

### BERT (training)

```python
from bedoner.models import bert_ner
from bedoner.ner_labels.labels_irex import ALL_LABELS
from bedoner.ner_labels.utils import make_biluo_labels
from spacy.util import minibatch

nlp=bert_ner(labels=make_biluo_labels(ALL_LABELS))
train_data=[["１９９９年３月創部の同部で初の外国人選手。", {"entities": [[0, 7, "DATE"], [15, 20, "ARTIFACT"]]}]]

niter=10
optim=nlp.resume_training(t_total=niter)
for i in range(niter):
    for batch in minibatch(train_data):
        texts, golds=zip(*batch)
        nlp.update(texts, golds,optim)
nlp(train_data[0][0]).ents
```
```
(１９９９年３月, 外国人選手)
```
```python
# save
nlp.to_disk("bert-foo")
# load
nlp = spacy.load("./bert-foo")
```

# Development

- 開発にあたり，spacyの仕組みについて公式ドキュメント([Architecture · spaCy API Documentation](https://spacy.io/api))を一読することをお勧めします．

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
