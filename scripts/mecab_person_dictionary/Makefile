#!/bin/sh
# python を指定したい場合は，PYTHON=$(which python3) make などとしてください
JINMEIDATA_TAR := "s3://camphr/jinmei_data.tar.gz"
MECAB_MODEL_PATH := "./jinmei_data/mecab-ipadic-2.7.0-20070801.model"
MECAB_IPADIC := "./jinmei_data/mecab-ipadic-2.7.0-20070801"
MECAB_DICT_INDEX_PATH := /usr/local/libexec/mecab/mecab-dict-index
ccend=$(echo -e "\033[0m")
USER_DICT_DIR := ${HOME}/.camphr

PY := $(if $(PYTHON),$(PYTHON),python3)

ifeq (, $(shell which $(PY)))
$(error $(PY) not found)
endif

ifeq (, $(MECAB_DICT_INDEX_PATH))
$(error $(MECAB_DICT_INDEX_PATH) not found)
endif

install: jinmei_data
	cd jinmei_data/mecab-ipadic-2.7.0-20070801 ; ./configure ; make
	$(PY) make_csv.py jinmei_data/jinmei30.txt jinmei_data/jinmei.csv
	mkdir -p $(USER_DICT_DIR)
	$(MAKE) create

create:
	$(MECAB_DICT_INDEX_PATH) -m $(MECAB_MODEL_PATH) -d $(MECAB_IPADIC) -u user.dic -f utf-8 -t utf-8 jinmei_data/jinmei.csv
jinmei_data.tar.gz:
	aws s3 cp $(JINMEIDATA_TAR) .
jinmei_data: jinmei_data.tar.gz
	tar xzvf jinmei_data.tar.gz
