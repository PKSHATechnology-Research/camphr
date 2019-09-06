#!/bin/sh
MECAB_MODEL_PATH="./mecab-ipadic-2.7.0-20070801.model"
MECAB_IPADIC="/usr/local/lib/mecab/dic/ipadic"
MECAB_DICT_INDEX_PATH="/usr/local/Cellar/mecab/0.996/libexec/mecab/mecab-dict-index"
$MECAB_DICT_INDEX_PATH -m $MECAB_MODEL_PATH -d $MECAB_IPADIC -u user.dic -f utf-8 -t utf-8 ./jinmei.csv
