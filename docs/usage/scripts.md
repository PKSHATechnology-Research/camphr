# test-package.sh

spacy packageでつくったpackageのテストスクリプトです．venvを作成し，その中でちゃんと動くかをテストします．  

```bash
$ scripts/test-package.sh foo.tar.gz
```

# versionup_package.py

model packageのバージョンをあげるスクリプトです．`bedoner`の依存バージョンを変更します．  
以前のリリースと同じモデルを再リリースするときに便利です．

```bash
$ scripts/test-package.sh foo.tar.gz
```