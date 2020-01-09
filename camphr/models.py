"""The models module defines functions to create spacy models."""
import os
from enum import Enum
from typing import Any, Dict

import camphr.lang.juman as juman
import camphr.lang.mecab as mecab
import camphr.trf_utils  # noqa: import to register optimizer
import spacy
from camphr.lang.torch_mixin import OPTIM_CREATOR
from camphr.pipelines.knp import KNP, juman_sentencizer
from camphr.pipelines.person_ner import create_person_ruler
from camphr.pipelines.trf_model import BertModel, XLNetModel
from camphr.pipelines.trf_ner import (
    BertForNamedEntityRecognition,
    TrfForNamedEntityRecognitionBase,
    XLNetForNamedEntityRecognition,
)
from camphr.pipelines.trf_seq_classification import (
    BertForSequenceClassification,
    XLNetForSequenceClassification,
)
from camphr.pipelines.trf_tokenizer import TransformersTokenizer
from spacy.language import Language
from spacy.pipeline import Sentencizer
from spacy.vocab import Vocab


def ja_sentencizer():
    return Sentencizer(Sentencizer.default_punct_chars + ["。"])


def knp() -> juman.Japanese:
    nlp = juman.Japanese()
    nlp.add_pipe(ja_sentencizer())
    nlp.add_pipe(juman_sentencizer)
    nlp.add_pipe(KNP.from_nlp(nlp))
    return nlp


def transformers_tokenizer(lang: str, pretrained: str) -> Language:
    meta: Dict[str, Any] = {OPTIM_CREATOR: "adamw"}
    if lang == "sentencepiece_torch":
        meta["tokenizer"] = {"model_path": pretrained}

    nlp = spacy.blank(lang, meta=meta)
    nlp.add_pipe(TransformersTokenizer.from_pretrained(str(pretrained)))
    return nlp


class TRF(Enum):
    bert = "bert"
    xlnet = "xlnet"


TRF_MODEL_MAP = {TRF.bert: BertModel, TRF.xlnet: XLNetModel}


def get_trf_name(pretrained: str) -> TRF:
    target = pretrained.lower()
    for name in TRF:
        if name.value in target:
            return name
    raise ValueError(f"Illegal pretrained name {pretrained}")


def trf_model(lang: str, pretrained: str, **cfg):
    nlp = transformers_tokenizer(lang, pretrained=pretrained)
    name = get_trf_name(pretrained)
    if name:
        model = TRF_MODEL_MAP[name].from_pretrained(pretrained, **cfg)
        nlp.add_pipe(model)
        return nlp


TRF_NER_MAP = {
    TRF.bert: BertForNamedEntityRecognition,
    TRF.xlnet: XLNetForNamedEntityRecognition,
}


def trf_ner(lang: str, pretrained: str, **cfg) -> Language:
    nlp = trf_model(lang, pretrained, **cfg)
    name = get_trf_name(pretrained)
    ner = TRF_NER_MAP[name].from_pretrained(nlp.vocab, pretrained, **cfg)
    nlp.add_pipe(ner)
    return nlp


def trf_ner_layer(
    lang: str, pretrained: str, vocab: Vocab, **cfg
) -> TrfForNamedEntityRecognitionBase:
    name = get_trf_name(pretrained)
    ner = TRF_NER_MAP[name].from_pretrained(vocab, pretrained, **cfg)
    return ner


TRF_SEQ_CLS_MAP = {
    TRF.bert: BertForSequenceClassification,
    TRF.xlnet: XLNetForSequenceClassification,
}


def trf_seq_classification(lang: str, pretrained: str, **cfg) -> Language:
    nlp = trf_model(lang, pretrained, **cfg)
    name = get_trf_name(pretrained)
    pipe = TRF_SEQ_CLS_MAP[name].from_pretrained(nlp.vocab, pretrained, **cfg)
    nlp.add_pipe(pipe)
    return nlp


def person_ruler(name="person_ruler") -> mecab.Japanese:
    user_dic = os.path.expanduser("~/.camphr/user.dic")
    if not os.path.exists(user_dic):
        raise ValueError(
            """User dictionary not found. See camphr/scripts/person_dictionary and create user dictionary."""
        )

    nlp = mecab.Japanese(
        meta={
            "tokenizer": {"userdic": user_dic, "assets": "./jinmei/"},
            "name": name,
            "requirements": ["mecab-python3"],
        }
    )
    nlp.add_pipe(create_person_ruler(nlp))
    return nlp
