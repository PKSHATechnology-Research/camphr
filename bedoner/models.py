import mojimoji
import json
import os
import spacy
from bedoner.lang.mecab import Japanese
from bedoner.entity_rulers.person import create_person_ruler
from pathlib import Path
from spacy.cli import package
from shutil import copy
from bedoner.entity_extractors.bert_modeling import BertModel
from bedoner.entity_extractors.bert_ner import BertEntityExtractor, create_estimator
from bedoner.lang.mecab import Japanese as Mecab
from bedoner.lang.juman import Japanese as Juman
from bedoner.lang.knp import Japanese as KNP
from pathlib import Path
from pathlib import Path
from spacy.strings import StringStore
from spacy.vocab import Vocab
from bedoner.wordpiecer import BertWordPiecer
from spacy.language import Language
import spacy
from bedoner.entity_rulers.date import DateRuler
import shutil


__dir__ = Path(__file__).parent


def bert_wordpiecer() -> Juman:
    with (__dir__ / "../data/Japanese_L-12_H-768_A-12_E-30_BPE/vocab.txt").open() as f:
        vs = []
        for line in f:
            vs.append(line[:-1])
    s = StringStore(vs)
    v = Vocab(strings=s)
    nlp = Juman(v, meta={"tokenizer": {"preprocessor": mojimoji.han_to_zen}})
    w = BertWordPiecer(
        v,
        vocab_file=str(__dir__ / "../data/Japanese_L-12_H-768_A-12_E-30_BPE/vocab.txt"),
    )
    w.model = w.Model(w.cfg["vocab_file"])
    nlp.add_pipe(w)
    return nlp


def bert_ner(name="bert_ner") -> Juman:
    nlp = bert_wordpiecer()
    nlp.meta["name"] = name

    bert_dir = __dir__ / "../data/Japanese_L-12_H-768_A-12_E-30_BPE"
    model_dir = __dir__ / "../data/bert_result_ene_0/"
    init_checkpoint = str(bert_dir / "bert_model.ckpt")
    with (model_dir / "label2id.json").open("r") as f:
        label2id = json.load(f)
    bert_cfg = dict(
        bert_dir=str(bert_dir),
        model_dir=str(model_dir),
        num_labels=len(label2id) + 1,
        init_checkpoint=init_checkpoint,
        use_one_hot_embeddings=None,
        max_seq_length=128,
        batch_size=10,
    )

    ee = BertEntityExtractor.from_nlp(nlp, label2id=label2id, **bert_cfg)
    ee.model = create_estimator(**bert_cfg)
    ee.set_values()
    ee.create_predictor()
    nlp.add_pipe(ee)
    return nlp


def date_ruler(name="date_ruler") -> Mecab:
    nlp = Mecab(meta={"name": "date_ruler", "requirements": ["mecab-python3", "regex"]})
    nlp.add_pipe(DateRuler(nlp))
    return nlp


def person_ruler(name="person_ruler") -> Mecab:
    user_dic = os.path.expanduser("~/.bedoner/user.dic")
    nlp = Japanese(
        meta={
            "tokenizer": {"userdic": user_dic, "assets": "./jinmei/"},
            "name": name,
            "requirements": ["mecab-python3", "regex"],
        }
    )
    nlp.add_pipe(create_person_ruler(nlp))
    return nlp
