## test-package.sh

spacy packageでつくったpackageのテストスクリプトです．venvを作成し，その中でちゃんと動くかをテストします．  

```bash
$ scripts/test-package.sh foo.tar.gz
```

## versionup_package.py

model packageのバージョンをあげるスクリプトです．`camphr`の依存バージョンを変更します．  
以前のリリースと同じモデルを再リリースするときに便利です．

```bash
$ scripts/test-package.sh foo.tar.gz
```

## packaging.py

`nlp.to_disk`を使って保存したモデルから，パッケージを作るスクリプトです．  
requrements等を設定したのち，`spacy.cli.package`に渡してpackage化し，`python setup.py sdist`を使ってtarballを作っています．

### example

```bash
$ python packaging.py model=foo/ version=v0.1.0

# help
$ python packaging.py --help

# show options
$ python packaging.py -c
```

## train.py

BERT NERをトレーニングするスクリプトです．今後トレーニング可能なタスクを増やしていく予定です．  

### example

```bash
$ python train.py lang=mecab label=irex

# help
$ python packaging.py --help

# show options
$ python packaging.py -c
```
