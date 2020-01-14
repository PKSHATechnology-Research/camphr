"""The models module defines functions to create spacy models."""
import os
from enum import Enum
from typing import Any, Dict, List

import camphr.lang.juman as juman
import camphr.lang.mecab as mecab
import camphr.pipelines.trf_utils  # noqa: import to register optimizer
from camphr.lang.torch_langs import get_torch_lang_cls
from camphr.lang.torch_mixin import OPTIM_CREATOR
from camphr.pipelines.knp import KNP, juman_sentencizer
from camphr.pipelines.person_ner import create_person_ruler
from camphr.pipelines.trf_model import TransformersModel
from camphr.pipelines.trf_ner import TrfForNamedEntityRecognition
from camphr.pipelines.trf_seq_classification import TrfForSequenceClassification
from camphr.pipelines.trf_tokenizer import TransformersTokenizer
from spacy.language import Language
from spacy.pipeline import Sentencizer


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
    cls = get_torch_lang_cls(lang)
    nlp = cls(meta=meta)
    nlp.add_pipe(TransformersTokenizer.from_pretrained(nlp.vocab, str(pretrained)))
    return nlp


class TRF(Enum):
    bert = "bert"
    xlnet = "xlnet"


def get_trf_name(pretrained: str) -> TRF:
    target = pretrained.lower()
    for name in TRF:
        if name.value in target:
            return name
    raise ValueError(f"Illegal pretrained name {pretrained}")


def trf_model(lang: str, pretrained: str, **cfg):
    nlp = transformers_tokenizer(lang, pretrained=pretrained)
    model = TransformersModel.from_pretrained(nlp.vocab, pretrained, **cfg)
    nlp.add_pipe(model)
    return nlp


def trf_ner(lang: str, pretrained: str, labels: List[str], **cfg) -> Language:
    nlp = trf_model(lang, pretrained, **cfg)
    ner = TrfForNamedEntityRecognition.from_pretrained(
        nlp.vocab, pretrained, labels=labels, **cfg
    )
    nlp.add_pipe(ner)
    return nlp


def trf_seq_classification(lang: str, pretrained: str, **cfg) -> Language:
    nlp = trf_model(lang, pretrained, **cfg)
    pipe = TrfForSequenceClassification.from_pretrained(nlp.vocab, pretrained, **cfg)
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
