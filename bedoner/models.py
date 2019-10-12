"""The models module defines functions to create spacy models."""
from bedoner.utils import inject_mixin
from typing import Type
from bedoner.ner_labels.labels_ontonotes import LANGUAGE
import os
from pathlib import Path

import bedoner.lang.juman as juman
import bedoner.lang.mecab as mecab
import bedoner.lang.knp as knp
import mojimoji
from bedoner.lang.trf_mixin import TransformersLanguageMixin
from bedoner.pipelines.date_ner import DateRuler
from bedoner.pipelines.person_ner import create_person_ruler
from bedoner.pipelines.trf_model import BertModel
from bedoner.pipelines.trf_ner import BertForNamedEntityRecognition
from bedoner.pipelines.knp_ner import KnpEntityExtractor
from bedoner.pipelines.wordpiecer import WordPiecer
from spacy.vocab import Vocab
from spacy.language import Language

__dir__ = Path(__file__).parent

bert_dir = str(__dir__ / "../data/bert-ja-juman")


def han_to_zen_normalizer(text):
    return mojimoji.han_to_zen(text.replace("\t", " ").replace("\r", ""))


def juman_nlp() -> juman.Japanese:
    return juman.Japanese(
        Vocab(), meta={"tokenizer": {"preprocessor": han_to_zen_normalizer}}
    )


def bert_wordpiecer(lang="juman", bert_dir=bert_dir) -> Language:
    if lang == "juman":
        cls = inject_mixin(TransformersLanguageMixin, juman.Japanese)
        nlp = cls(Vocab(), meta={"tokenizer": {"preprocessor": han_to_zen_normalizer}})
    elif lang == "mecab":
        cls = inject_mixin(TransformersLanguageMixin, mecab.Japanese)
        nlp = cls(Vocab())
    else:
        raise ValueError(f"Unsupported lang: {lang}")
    w = WordPiecer.from_pretrained(nlp.vocab, bert_dir)
    nlp.add_pipe(w)
    return nlp


def bert_model(lang="juman", bert_dir=bert_dir):
    nlp = bert_wordpiecer(lang)
    bert = BertModel.from_pretrained(Vocab(), bert_dir)
    nlp.add_pipe(bert)
    return nlp


def bert_ner(lang="juman", bert_dir=bert_dir, **cfg):
    nlp = bert_model(lang)
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
