# NER統合 ![CircleCI](https://circleci.com/gh/PKSHATechnology/bedore-ner.svg?style=svg)

# Installation

## requrements

- mecab, juman(pp), knpが必要です．

## 完成品

- [release page](https://github.com/PKSHATechnology/bedore-ner/releases)からtar.gzをダウンロードしてください
- `pip install foo.tar.gz`でOKです．
	- 必要な辞書，パラメータ等全てセットアップされます．

### 例

```
$ pip install mecab_entity_ruler-0.0.0.tar.gz
```
```python
>> import spacy
>> nlp=spacy.load("mecab_entity_ruler")
>> nlp("2019年11月8日に高松隆と東京タワーに行った").ents
(2019年11月8日, 高松隆)
```

## Refs.

- attlassian 
	- https://pkshatech.atlassian.net/wiki/spaces/CHAT/pages/35061918/NER+2019+8
- NERのタグ
	- https://gist.github.com/kzinmr/14c224efc43b7e21ff95fa9c54f829f1
