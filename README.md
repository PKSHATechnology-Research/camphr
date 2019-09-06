# NER統合 ![CircleCI](https://circleci.com/gh/PKSHATechnology/bedore-ner.svg?style=svg)

# Installation

## requrements

- mecab, juman(pp), knpが必要です．

## 完成品

- release pageからtar.gzをダウンロードしてください
- `pip install foo.tar.gz`でOKです．
	- person_rulerにいては，MeCabの辞書がコンパイルされるのでちょっと時間がかかります．

```
>> import spacy
>> nlp=spacy.load("knp_entity_extractor")
>> nlp("2019年11月8日に高松隆と東京タワーに行った").ents
(2019年11月8日, 高松隆, 東京タワー)
```

## Refs.

- attlassian 
	- https://pkshatech.atlassian.net/wiki/spaces/CHAT/pages/35061918/NER+2019+8
- NERのタグ
	- https://gist.github.com/kzinmr/14c224efc43b7e21ff95fa9c54f829f1
