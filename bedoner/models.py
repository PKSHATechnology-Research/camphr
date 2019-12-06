"""The models module defines functions to create spacy models."""
import os
from enum import Enum
from typing import Any, Dict

import bedoner.lang.juman as juman
import bedoner.lang.knp as knp
import bedoner.lang.mecab as mecab
import bedoner.lang.sentencepiece as sp
import bedoner.trf_utils  # noqa: import to register optimizer
from bedoner.lang.torch_mixin import OPTIM_CREATOR
from bedoner.pipelines.knp_ner import KnpEntityExtractor
from bedoner.pipelines.person_ner import create_person_ruler
from bedoner.pipelines.trf_model import BertModel, XLNetModel
from bedoner.pipelines.trf_ner import (
    BertForNamedEntityRecognition,
    TrfForNamedEntityRecognitionBase,
    XLNetForNamedEntityRecognition,
)
from bedoner.pipelines.trf_seq_classification import (
    BertForSequenceClassification,
    XLNetForSequenceClassification,
)
from bedoner.pipelines.wordpiecer import TrfSentencePiecer, WordPiecer
from spacy.language import Language
from spacy.vocab import Vocab


def han_to_zen_normalizer(text):
    try:
        import mojimoji
    except ImportError:
        raise ValueError("juman or knp Language requires mojimoji.")
    return mojimoji.han_to_zen(text.replace("\t", " ").replace("\r", ""))


def juman_nlp() -> juman.Japanese:
    return juman.Japanese(
        Vocab(), meta={"tokenizer": {"preprocessor": han_to_zen_normalizer}}
    )


def wordpiecer(lang: str, pretrained: str) -> Language:
    meta: Dict[str, Any] = {OPTIM_CREATOR: "adamw"}
    if lang == "juman":
        cls = juman.TorchJapanese
        meta["tokenizer"] = {"preprocessor": han_to_zen_normalizer}
    elif lang == "mecab":
        cls = mecab.TorchJapanese
    elif lang == "sentencepiece":
        cls = sp.TorchSentencePieceLang
        meta["tokenizer"] = {"model_path": pretrained}
    else:
        raise ValueError(f"Unsupported lang: {lang}")

    nlp = cls(Vocab(), meta=meta)

    if lang == "sentencepiece":
        w = TrfSentencePiecer.from_pretrained(nlp.vocab, pretrained)
    else:
        w = WordPiecer.from_pretrained(nlp.vocab, pretrained)
    nlp.add_pipe(w)
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
    nlp = wordpiecer(lang, pretrained=pretrained)
    name = get_trf_name(pretrained)
    if name:
        model = TRF_MODEL_MAP[name].from_pretrained(nlp.vocab, pretrained, **cfg)
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
