# ユーザ辞書作成手順
## 前準備
https://pkshatech.atlassian.net/wiki/spaces/~n_sumino/pages/15073573/MeCab

上記のコンフルドキュメントに従って, mecab-ipadic/mecab-ipadic-modelをbedore-ner-module/scripts/user_dictionary下に配置・設定しておく。

## 辞書作成手順
特定の単語を確実に名前として認識させたい場合は
- `JINMEI30.csv` に要素を追加
- `create_user_dictionary.py` の `NAMES_TO_BALANCE` に当該苗字・名前を追加する
- `sh create_user_dictionary.sh` を実行
