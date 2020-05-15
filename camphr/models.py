"""`models` defined `camphr.load`, camphr model loader."""
import functools
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union, cast

import omegaconf
import spacy
import toolz
from omegaconf import OmegaConf
from spacy.language import Language
from spacy.pipeline import Pipe
from toolz import merge
from typing_extensions import Literal

from camphr.lang.torch import TorchLanguage
from camphr.ner_labels.utils import get_ner_labels
from camphr.pipelines.transformers.model import TRANSFORMERS_MODEL
from camphr.pipelines.transformers.ner import TRANSFORMERS_NER
from camphr.pipelines.transformers.seq_classification import (
    TRANSFORMERS_MULTILABEL_SEQ_CLASSIFIER,
    TRANSFORMERS_SEQ_CLASSIFIER,
)
from camphr.pipelines.transformers.tokenizer import TRANSFORMERS_TOKENIZER
from camphr.pipelines.transformers.utils import LABELS
from camphr.utils import get_labels, resolve_alias

__dir__ = Path(__file__).parent
_MODEL_CFG_DIR = __dir__ / "model_config"
_PREDEFINED_CONFS = {"knp": _MODEL_CFG_DIR / "knp.yml"}
CFG_SRC = Union[str, Dict[str, Any]]


def load(cfg: Union[str, CFG_SRC]) -> Language:
    if isinstance(cfg, str) and cfg in _PREDEFINED_CONFS:
        cfg = _PREDEFINED_CONFS[cfg].read_text()
    return create_model(cfg)


__all__ = ["load"]


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
    task: Optional[Literal["ner", "textcat", "multilabel_textcat"]]
    labels: Optional[str]


def create_model(cfg: Union[NLPConfig, Any]) -> Language:
    if not isinstance(cfg, omegaconf.Config):
        cfg = OmegaConf.create(cfg)
    cfg = correct_model_config(cfg)
    nlp = create_lang(cfg.lang)
    for pipe in create_pipeline(nlp, cfg.pipeline):
        nlp.add_pipe(pipe)
    if cfg.name and isinstance(cfg.name, str):
        nlp._meta["name"] = cfg.name
    nlp._meta["config"] = OmegaConf.to_container(cfg.lang)
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
        return TorchLanguage(True, optimizer_config=cfg.optimizer, **kwargs)
    return spacy.blank(cfg.name, **kwargs)


def create_pipeline(nlp: Language, cfg: omegaconf.DictConfig) -> List[Pipe]:
    if not isinstance(cfg, omegaconf.DictConfig):
        cfg = OmegaConf.create(cfg)

    pipes = []
    for name, pipe_config in cfg.items():
        pipe_config = OmegaConf.to_container(pipe_config or OmegaConf.create({}))
        pipes.append(nlp.create_pipe(name, config=pipe_config or dict()))
    return pipes


_ConfigParser = Callable[[NLPConfig], NLPConfig]


def correct_model_config(cfg: NLPConfig) -> NLPConfig:
    """Parse config. Complement missing informations, resolve aliases, etc."""
    PARSERS: List[_ConfigParser] = [
        resolve_alias(ALIASES),
        _add_pipes,
        _add_required_pipes,
        _align_pipeline,
        _correct_trf_pipeline,
        _resolve_label,
    ]
    return toolz.pipe(cfg, *PARSERS)


# Alias definition.
# For example, `pretrained` key is converted to `pipeline.transformers_model.trf_name_or_path`.
# The aliases in config is resolved by `resolved_alias`
ALIASES = {
    "pretrained": f"pipeline.{TRANSFORMERS_MODEL}.trf_name_or_path",
    "ner_label": f"pipeline.{TRANSFORMERS_NER}.labels",
    "textcat_label": f"pipeline.{TRANSFORMERS_SEQ_CLASSIFIER}.labels",
    "multitextcat_label": f"pipeline.{TRANSFORMERS_MULTILABEL_SEQ_CLASSIFIER}.labels",
    "optimizer": "lang.optimizer",
}


TRF_BASES = [TRANSFORMERS_TOKENIZER, TRANSFORMERS_MODEL]
TRF_TASKS = [
    TRANSFORMERS_SEQ_CLASSIFIER,
    TRANSFORMERS_NER,
    TRANSFORMERS_MULTILABEL_SEQ_CLASSIFIER,
]
TRF_PIPES = TRF_BASES + TRF_TASKS


# Mapping for pipeline requirements.
# {"pipename": [required pipe names in front]}
# e.g. `TRANSFORMERS_MODEL` requires `TRANSFORMERS_TOKENIZER` in front.
PIPELINE_ALIGNMENT = {
    TRANSFORMERS_MODEL: [TRANSFORMERS_TOKENIZER],
    TRANSFORMERS_NER: [TRANSFORMERS_MODEL],
    TRANSFORMERS_SEQ_CLASSIFIER: [TRANSFORMERS_MODEL],
    TRANSFORMERS_MULTILABEL_SEQ_CLASSIFIER: [TRANSFORMERS_MODEL],
}

TASK2PIPE = {
    "textcat": f"{TRANSFORMERS_SEQ_CLASSIFIER}",
    "ner": f"{TRANSFORMERS_NER}",
    "multilabel_textcat": f"{TRANSFORMERS_MULTILABEL_SEQ_CLASSIFIER}",
}


TASKS = {"textcat", "ner", "multilabel_textcat"}


def _add_pipes(cfg: NLPConfig) -> NLPConfig:
    if cfg.task in TASKS:
        assert cfg.labels, "`cfg.labels` required"
        cfg.pipeline = cfg.pipeline or OmegaConf.create({})
        # TODO https://github.com/microsoft/pyright/issues/671
        pipe = TASK2PIPE[cast(str, cfg.task)]
        prev = cfg.pipeline[pipe] or OmegaConf.create({})
        cfg.pipeline[pipe] = OmegaConf.merge(
            OmegaConf.create({"labels": cfg.labels}), prev
        )
    else:
        assert not cfg.labels, f"One of {TASKS} pipeline is required."
    return cfg


def _add_required_pipes(cfg: NLPConfig) -> NLPConfig:
    """assign required pipes defined in  `PIPELINE_ALIGNMENT`"""
    pipe_names = list(cfg.pipeline.keys())
    while True:
        prev_len = len(pipe_names)
        for k, v in PIPELINE_ALIGNMENT.items():
            for vv in v:
                if k in pipe_names and (vv not in pipe_names):
                    pipe_names.insert(pipe_names.index(k), vv)
        if prev_len == len(pipe_names):
            break
    cfg.pipeline = {k: cfg.pipeline.get(k, {}) for k in pipe_names}
    return cfg


def _align_pipeline(cfg: NLPConfig) -> NLPConfig:
    """align pipelines based on `PIPELINE_ALIGNMENT`"""
    pipe_names = sorted(cfg.pipeline.keys(), key=_pipe_cmp_key)
    cfg.pipeline = {k: cfg.pipeline.get(k, {}) for k in pipe_names}
    return cfg


@functools.cmp_to_key
def _pipe_cmp_key(x: str, y: str) -> int:
    """key function for `_align_pipeline`"""
    if _is_ancestor(x, y):
        return 1  # x comes after y
    if _is_ancestor(y, x):
        return -1  # y comes after x
    return 0  # no order relation to x and y


def _is_ancestor(child: str, ancestor: str) -> bool:
    if child not in PIPELINE_ALIGNMENT:
        return False
    parents = deque(PIPELINE_ALIGNMENT[child])
    while parents:
        p = parents.popleft()
        if p == ancestor:
            return True
        parents.extend(PIPELINE_ALIGNMENT.get(p, []))
    return False


def _correct_trf_pipeline(cfg: NLPConfig) -> NLPConfig:
    """Correct config of transformers pipeline

    Note:
        1. Complement `trf_name_or_path` for transformers pipelines.
        2. If there are trf pipe in pipeline, set lang.torch = true
    """
    cfg = _complement_trf_name(cfg)
    cfg = _correct_torch(cfg)
    return cfg


def _complement_trf_name(cfg: NLPConfig) -> NLPConfig:
    """For transformers pipeline.

    All transformers pipeline requires same `trf_name_or_path`, but it is a pain to write down.
    So if there is any transformers pipeline with `trf_name_or_path`, copy to other transformers pipes.
    """
    KEY = "trf_name_or_path"
    VAL = ""
    if not set(cfg.pipeline.keys()) & set(TRF_PIPES):
        return cfg
    for k, v in cfg.pipeline.items():
        VAL = v[KEY] or VAL
    if not VAL:
        raise ValueError(
            f"Invalid configuration. At least one of transformer's pipe needs `{KEY}`, but the configuration is:\n"
            + cfg.pipeline.pretty()
        )
    for k, v in cfg.pipeline.items():
        if k in TRF_PIPES and not v.get(KEY, None):
            v[KEY] = VAL
    return cfg


def _correct_torch(cfg: NLPConfig) -> NLPConfig:
    """If there is any transformers pipeline, set `torch` to True"""
    if set(cfg.pipeline) & set(TRF_PIPES):
        cfg.lang.torch = True
    return cfg


def _resolve_label(cfg: NLPConfig) -> NLPConfig:
    """Resolver for ner and sequence classification label config."""
    ner = cfg.pipeline[TRANSFORMERS_NER]
    if ner:
        ner[LABELS] = get_ner_labels(ner[LABELS])
    seq = cfg.pipeline[TRANSFORMERS_SEQ_CLASSIFIER]
    if seq:
        seq[LABELS] = get_labels(seq[LABELS])
    multiseq = cfg.pipeline[TRANSFORMERS_MULTILABEL_SEQ_CLASSIFIER]
    if multiseq:
        multiseq[LABELS] = get_labels(multiseq[LABELS])
    return cfg
