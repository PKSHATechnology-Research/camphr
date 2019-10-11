"""The models module defines functions to create spacy models."""
import os
from pathlib import Path

import bedoner.lang.juman as juman
import bedoner.lang.mecab as mecab
import bedoner.lang.knp as knp
import mojimoji
from bedoner.lang.trf_mixin import TransformersJuman
from bedoner.pipelines.date_ner import DateRuler
from bedoner.pipelines.person_ner import create_person_ruler
from bedoner.pipelines.trf_model import BertModel
from bedoner.pipelines.trf_ner import BertForNamedEntityRecognition
from bedoner.pipelines.knp_ner import KnpEntityExtractor
from bedoner.pipelines.wordpiecer import WordPiecer
from spacy.vocab import Vocab

__dir__ = Path(__file__).parent

bert_dir = str(__dir__ / "../data/bert-ja-juman")


def han_to_zen_normalizer(text):
    return mojimoji.han_to_zen(text.replace("\t", " ").replace("\r", ""))


def juman_nlp() -> juman.Japanese:
    return juman.Japanese(
        Vocab(), meta={"tokenizer": {"preprocessor": han_to_zen_normalizer}}
    )


def bert_wordpiecer() -> mecab.Japanese:
    nlp = TransformersJuman(
        Vocab(), meta={"tokenizer": {"preprocessor": han_to_zen_normalizer}}
    )
    w = WordPiecer.from_pretrained(Vocab(), bert_dir)
    nlp.add_pipe(w)
    return nlp


def bert_model():
    nlp = bert_wordpiecer()
    bert = BertModel.from_pretrained(Vocab(), bert_dir)
    nlp.add_pipe(bert)
    return nlp


def bert_ner(**cfg):
    nlp = bert_model()
    ner = BertForNamedEntityRecognition.from_pretrained(Vocab(), bert_dir, **cfg)
    nlp.add_pipe(ner)
    return nlp


def date_ruler(name="date_ruler") -> mecab.Japanese:
    nlp = mecab.Japanese(
        meta={"name": name, "requirements": ["mecab-python3", "regex"]}
    )
    nlp.add_pipe(DateRuler())
    return nlp


def person_ruler(name="person_ruler") -> mecab.Japanese:
    user_dic = os.path.expanduser("~/.bedoner/user.dic")
    if not os.path.exists(user_dic):
        raise ValueError(
            """User dictionary not found. See bedoner/scripts/person_dictionary and create user dictionary."""
        )

    nlp = mecab.Japanese(
        meta={
            "tokenizer": {"userdic": user_dic, "assets": "./jinmei/"},
            "name": name,
            "requirements": ["mecab-python3", "regex"],
        }
    )
    nlp.add_pipe(create_person_ruler(nlp))
    return nlp


def knp_ner(name="knp_ner") -> knp.Japanese:
    nlp = knp.Japanese(meta={"name": name, "requirements": ["pyknp"]})
    nlp.add_pipe(KnpEntityExtractor(nlp))
    return nlp
