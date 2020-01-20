"""The models module defines functions to create spacy models."""
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import omegaconf
import spacy
from cytoolz import merge
from omegaconf import OmegaConf
from spacy.language import Language
from spacy.vocab import Vocab

from camphr.lang.torch import TorchLanguage
from camphr.ner_labels.utils import get_biluo_labels
from camphr.pipelines.transformers.model import TRANSFORMERS_MODEL
from camphr.pipelines.transformers.ner import TRANSFORMERS_NER
from camphr.pipelines.transformers.seq_classification import TRANSFORMERS_SEQ_CLASSIFIER
from camphr.pipelines.transformers.tokenizer import TRANSFORMERS_TOKENIZER
from camphr.pipelines.transformers.utils import LABELS
from camphr.utils import create_dict_from_dotkey, get_by_dotkey, get_labels

__dir__ = Path(__file__).parent
_MODEL_CFG_DIR = __dir__ / "model_config"
_PREDEFINED_CONFS = {"knp": _MODEL_CFG_DIR / "knp.yml"}
CFG_SRC = Union[str, Dict[str, Any]]


def load(cfg_or_name: Union[str, CFG_SRC]) -> Language:
    if isinstance(cfg_or_name, str) and cfg_or_name in _PREDEFINED_CONFS:
        cfg_or_name = _PREDEFINED_CONFS[cfg_or_name].read_text()
    return create_model(cfg_or_name)


@dataclass
class LangConfig(omegaconf.Config):
    name: str
    torch: bool
    optimizer: Dict[str, Any]
    kwargs: Optional[Dict[str, Any]]


@dataclass
class NLPConfig(omegaconf.Config):
    name: str
    lang: LangConfig
    pipeline: omegaconf.DictConfig


def create_model(cfg: Union[NLPConfig, Any]) -> Language:
    if not isinstance(cfg, omegaconf.Config):
        cfg = OmegaConf.create(cfg)
    cfg = correct_model_config(cfg)
    nlp = create_lang(cfg.lang)
    for name, config in cfg.pipeline.items():
        if config:
            config = OmegaConf.to_container(config)
        nlp.add_pipe(nlp.create_pipe(name, config=config or dict()))
    if cfg.name and isinstance(cfg.name, str):
        nlp._meta["name"] = cfg.name
    return nlp


def create_lang(cfg: LangConfig) -> Language:
    kwargs = cfg.kwargs or {}
    kwargs = (
        OmegaConf.to_container(kwargs)
        if isinstance(kwargs, omegaconf.Config)
        else kwargs
    )
    if cfg.torch:
        kwargs["meta"] = merge(kwargs.get("meta", {}), {"lang": cfg.name})
        return TorchLanguage(Vocab(), optimizer_config=cfg.optimizer, **kwargs)
    return spacy.blank(cfg.name, **kwargs)


def correct_model_config(cfg: NLPConfig) -> NLPConfig:
    cfg = _resolve_alias(cfg)
    cfg = _align_pipeline(cfg)
    cfg = _correct_trf_pipeline(cfg)
    cfg = _resolve_label(cfg)
    return cfg


def _correct_trf_pipeline(cfg: NLPConfig) -> NLPConfig:
    """Correct config for transformers pipeline

    Note:
        1. Complement `trf_name_or_path` for transformers pipelines.
        2. If there are trf pipe in pipeline, set lang.torch = true
    """
    cfg = _complement_trf_name(cfg)
    cfg = _correct_torch(cfg)
    return cfg


TRF_BASES = [TRANSFORMERS_TOKENIZER, TRANSFORMERS_MODEL]
TRF_TASKS = [TRANSFORMERS_SEQ_CLASSIFIER, TRANSFORMERS_NER]
TRF_PIPES = TRF_BASES + TRF_TASKS

ALIASES = {
    "pretrained": f"pipeline.{TRANSFORMERS_MODEL}.trf_name_or_path",
    "ner_label": f"pipeline.{TRANSFORMERS_NER}.labels",
    "textcat_label": f"pipeline.{TRANSFORMERS_SEQ_CLASSIFIER}.labels",
}


def _resolve_alias(cfg: NLPConfig) -> NLPConfig:
    for alias, name in ALIASES.items():
        v = get_by_dotkey(cfg, alias)
        if v is None:
            continue
        cfg = OmegaConf.merge(cfg, OmegaConf.create(create_dict_from_dotkey(name, v)))
    return cfg


# Mapping for pipeline alignment.
# pipename: required pipe names in front
# e.g. `TRANSFORMERS_MODEL` requires `TRANSFORMERS_TOKENIZER` in front.
# Why `required pipe` is list? - For depending on multiple pipes.
PIPELINE_ALIGNMENT = {
    TRANSFORMERS_MODEL: [TRANSFORMERS_TOKENIZER],
    TRANSFORMERS_NER: [TRANSFORMERS_MODEL],
    TRANSFORMERS_SEQ_CLASSIFIER: [TRANSFORMERS_MODEL],
}


def _align_pipeline(cfg: NLPConfig) -> NLPConfig:
    pipe_names = list(cfg.pipeline.keys())

    def _align(pipe_names):
        for p in set(PIPELINE_ALIGNMENT) & set(pipe_names):
            for q in PIPELINE_ALIGNMENT[p]:
                pi = pipe_names.index(p)
                if q not in pipe_names:
                    pipe_names.insert(pi, q)
                else:
                    qi = pipe_names.index(q)
                    if pi < qi:
                        pipe_names[pi], pipe_names[qi] = pipe_names[qi], pipe_names[pi]
        return pipe_names

    while True:
        prev = copy.copy(pipe_names)
        pipe_names = _align(pipe_names)
        if prev == pipe_names:
            break

    cfg.pipeline = {k: cfg.pipeline.get(k, {}) for k in pipe_names}
    return cfg


def _complement_trf_name(cfg: NLPConfig) -> NLPConfig:
    KEY = "trf_name_or_path"
    VAL = ""
    if not set(cfg.pipeline.keys()) & set(TRF_PIPES):
        return cfg
    for k, v in cfg.pipeline.items():
        if k in TRF_PIPES and v[KEY]:
            VAL = v[KEY]
    if not VAL:
        raise ValueError(
            f"Invalid configuration. At least one of transformer's pipe needs `{KEY}`, but the configuration is:\n"
            + cfg.pipeline.pretty()
        )
    for k, v in cfg.pipeline.items():
        if k in TRF_PIPES:
            v[KEY] = VAL
    return cfg


def _resolve_label(cfg: NLPConfig) -> NLPConfig:
    ner = cfg.pipeline[TRANSFORMERS_NER]
    if ner:
        ner[LABELS] = get_biluo_labels(ner[LABELS])
    seq = cfg.pipeline[TRANSFORMERS_SEQ_CLASSIFIER]
    if seq:
        seq[LABELS] = get_labels(seq[LABELS])
    return cfg


def _correct_torch(cfg: NLPConfig) -> NLPConfig:
    if set(cfg.pipeline) & set(TRF_PIPES):
        cfg.lang.torch = True
    return cfg
