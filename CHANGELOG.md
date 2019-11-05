# v0.4

##  New features and improvements

- XLNet追加
  - `trf_model.XLNetModel`を追加
  - `trf_ner.XLNetForNamedEntityRecognition`を追加
- SentencePiece ベースの`Language`を追加: [./bedoner/lang/sentencepiece](./bedoner/lang/sentencepiece)
- SentencePiece Language用のWordpiecer: `pipelines.wordpiecer.TrfSentencePiecer`を追加
- テストの高速化

- Multitask NER modelを追加
  - https://github.com/PKSHATechnology/bedore-ner/releases/tag/v0.4.0.dev3
  - ENEラベルについて，3つの異なるラベルづけモデルを用いてNER.
    - 例: Name/Organization/Familyの場合,"Name"と"Organization"と"Family"を予測する3種のモデルを作る．学習はマルチタスク．抽出結果が衝突した場合，下位階層予測のモデルが優先される
- `TrfForNamedEntityRecognitionBase`にuser_hooksを追加
  - goldラベルを適当に変形したい場合，`ner.user_hooks["convert_label"] = fn`のようにする
- `bedoner.pipelines.regex_pipe.MultipleRegexRuler`を追加
  - 複数パターンを登録できる正規表現パイプです


## Backwards incompatibilities

- `trf_ner.BertForTokenClassification`を`trf_ner.TrfForTOkenClassification`へ変更
- `trf_ner.BertTokenClassifier`を`trf_ner.TrfTokenClassifier`へ変更

# v0.3.1

##  New features and improvements

- multilingual BERT (`bert-base-multilingual-cased`)を使用したNERモデル [mecab_bert_multilingual_ner をリリース](https://github.com/PKSHATechnology/bedore-ner/releases/tag/v0.3.1.dev0)しました．
- `bedoner.models.bert_model`を改善し，transformersモデルを簡単に使用できるようにしました．
  - ex) `nlp = bert_model(lang=mecab, pretrained="bert-base-multilingual-cased")`

# v0.3

milestone: https://github.com/PKSHATechnology/bedore-ner/milestone/1?closed=1

##  New features and improvements

- BERTベースのモデルを[新たに6種類リリース](https://github.com/PKSHATechnology/bedore-ner/releases/tag/v0.3.0.dev2)
  - mecab ner (ene, irex)
  - juman ner (ene, irex)
  - mecab, juman pretrained model
- BERT modelについて, 埋め込みベクトル`vector`およびコサイン類似度`similarity`機能を追加
  - 使い方: [docs/usage/pipelines.md#BERT](./docs/usage/pipelines.md#BERT)
- mecab, jumanの依存を外しました (#44)
  - これらに依存する機能を使いたい場合，個別にインストールする必要があります．
- gold dataの分割スクリプトを追加 (参考: https://github.com/PKSHATechnology/bedore-ner/blob/version%2Fv0.3/docs/usage/cli.md)
- model packagingのversionup scriptを追加 [scripts/versionup-package](scripts/versionup-package)(#60)
- model package用のテストスクリプトを追加([scripts/test-package.sh](./scripts/test-package.sh)) (#9)
- NERのラベルづけがoverlapする際に有用なutil `pipelines.utils.merge_entities`を追加 (#63)
 
## Bug fixes

- trf nerに全角スペースが入力エラーにならないようにしました (#15)
- mecabについて，全角スペースを半角スペースと同様に扱うようにしました (#39)

## Documentation and examples

- [docs](./docs)ディレクトリにドキュメントをいくつか追加．
  - 概要: [docs/README.md](./docs/README.md)

## Refactor

- `pipelines.date_ner`をregex_rulerに統合(#61)
- scripts directoryを整理
- removed MatcherRuler (#62)
  - spacyの`EntityRuler`を使いましょう．

# v0.2

- change dependency
  - pytorch-transformers -> transformers
  - spacy-pytorch-transformers - > pytorch-transformers
- gpu support

```
nlp.to(torch.device("cuda"))
```

- python3.6 conpatibility
- training script: scripts/train.py

# v0.1.1

- mecabについて，urlを1トークンとして扱うようにした (#42)
- regex_ruler の追加 (#43)
- postcoder ruler の追加 (#43)
- matcher_ruler の追加 (#45)
- person_ner, date_ner について，LABELをexport
- `bedoner.__version__` の追加

# v0.1

- mecab, juman, knp Language について，スペースの取り扱いを改善
  - tokenにスペースは含めないが，`doc.text`には含まれる