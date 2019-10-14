# pipelines

## BERT NER

[BERT](https://github.com/google-research/bert)を用いたNERです．  
[リリースページ](https://github.com/PKSHATechnology/bedore-ner/releases)からトレーニング済みモデルをダウンロードして，以下のようにpipでインストールしてください.  
パラメータ等全て入っています．

```bash
$ pip install mecab-bert-ene.VERSION.tar.gz
```

```python
>>> import spacy
>>> nlp = spacy.load("mecab_bert_ene")
>>> doc = nlp("10日発表されたノーベル文学賞の受賞者をめぐり、選考機関のスウェーデン・アカデミーが批判されている。")
>>> for e in doc.ents:
>>>     print(e.text, e.label_)
10日 DATE
ノーベル文学賞 AWARD
スウェーデン COUNTRY
```

大量の入力を処理したいときは，`nlp.pipe`を使いましょう．遅延評価であり，無限長の入力にも対応しています．

```python
>>> texts: Iterable[str] = ...
>>> docs = nlp(texts)
>>> for doc in docs:
>>>   ...
```

GPUを使うことでさらに高速に処理できます．(内部ではpytorchを使用しています)

```python
>>> import torch
>>> nlp.to(torch.device("gpu"))
>>> docs = nlp(texts)
```

### Training

Pretrained モデルをさらにトレーニングすることもできます．  
以下のように，`nlp.update`にデータを与えるだけでOKです．

```python
from bedoner.models import bert_ner
from bedoner.ner_labels.labels_irex import ALL_LABELS
from bedoner.ner_labels.utils import make_biluo_labels
from spacy.util import minibatch

nlp = bert_ner(labels=make_biluo_labels(ALL_LABELS))
train_data = [["１９９９年３月創部の同部で初の外国人選手。", {"entities": [[0, 7, "DATE"], [15, 20, "ARTIFACT"]]}]]

niter = 10
optim = nlp.resume_training(t_total=niter)
for i in range(niter):
    for batch in minibatch(train_data):
        texts, golds = zip(*batch)
        nlp.update(texts, golds,optim)
nlp(train_data[0][0]).ents
```
```
(１９９９年３月, 外国人選手)
```

トレーニングが済んだモデルは, `nlp.to_disk`で保存して再利用できます．

```python
# save
nlp.to_disk("bert-foo")
# load
nlp = spacy.load("./bert-foo")
```

## Regex NER

`bedoner.pipelines.RegexRuler`を使うと，正規表現を用いたNER pipeを作ることができます．  
例えば，電話番号を検出したい場合は以下のようにします．  

```python
import spacy
from bedoner.pipelines import RegexRuler

nlp = spacy.blank("mecab")
# create pipe
pipe = RegexRuler(pattern="\d{3}-\d{4}-\d{4}", label="PHONE")
nlp.add_pipe(pipe)

text = "もし用があれば080-1234-7667にかけてください"
nlp(text).ents
```
```
(080-1234-7667,)
```

### Compose with BERT

BERTと正規表現pipeを組み合わせて使うこともできます．特定の表現についてrecallを100%にしたいときなどに有用です．  
例えばルールベースの電話番号検出をBERTに加える場合，以下の通りです．  

```python
import spacy
nlp = spacy.load("mecab_bert_ene")
doc = nlp("10日発表されたノーベル文学賞の受賞者をめぐり、選考機関のスウェーデン・アカデミーが批判されている。")
for e in doc.ents:
    print(e.text, e.label_)
```


### Built-In regex pipes

以下のpipeは，パッケージに同梱されています．

- postcode
- carcode

Example: 

```python
import spacy
from bedoner.pipelines import postcode_ruler, carcode_ruler

nlp = spacy.blank("mecab")
nlp.add_pipe(postcode_ruler)
nlp.add_pipe(carcode_ruler)

nlp("郵便番号は〒100-0001で，車の番号は品川500 さ 2345です").ents
```
```
(〒100-0001, 品川500 さ 2345)
```

### Advanced

recallを100%にしたい場合は，`pipe.destructive = True`にします．分かち書きで作成したtokenを分解し，確実にマッチするようになりますが，他のパイプの性能を落とす可能性があります．

## Person NER

mecabのタグ情報を元に，人名抽出をします．

```python
>>> from bedoner.models import person_ruler
>>> nlp = person_ruler()
>>> text = "2019年11月8日に高松隆と東京タワーに行った"
>>> nlp(text).ents
(高松隆,)
```

## Date NER

ルールベースで日付を抽出するNERパイプラインです．

```python
>>> from bedoner.models import date_ruler
>>> nlp = date_ruler()
>>> text = "2019年11月8日に高松隆と東京タワーに行った"
>>> nlp(text).ents
(2019年11月8日,)
```

## KNP NER

[KNP](http://nlp.ist.i.kyoto-u.ac.jp/index.php?KNP)を使ったNERです．

```python
>>> from bedoner.models import knp_ner
>>> nlp = knp_ner()
>>> text = "2019年11月8日に高松隆と東京タワーに行った"
>>> nlp(text).ents
(2019年11月8日, 高松隆, 東京タワー)
```
