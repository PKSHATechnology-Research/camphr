"""`models` defined `camphr.load`, camphr model loader.

TODO: strictly type
"""
from collections import deque
from dataclasses import dataclass
import dataclasses
import functools
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, Union, cast
from typing_extensions import Literal

import dataclass_utils
import spacy
from spacy.language import Language
from spacy.pipeline import Pipe
from toolz import merge

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
from camphr.utils import get_labels, resolve_alias, yaml_to_dict

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
class LangConfig:
    name: str
    torch: bool = False
    optimizer: Optional[Dict[str, Any]] = None
    kwargs: Optional[Dict[str, Any]] = None


@dataclass
class NLPConfig:
    lang: LangConfig
    pipeline: Dict[str, Optional[Dict[str, Any]]] = dataclasses.field(
        default_factory=dict
    )
    task: Optional[Literal["ner", "textcat", "multilabel_textcat"]] = None
    labels: Optional[Union[str, List[str]]] = None
    name: Optional[str] = None
    pretrained: Optional[str] = None
    ner_label: Optional[List[str]] = None
    textcat_label: Optional[List[str]] = None
    multitextcat_label: Optional[List[str]] = None
    optimizer: Optional[Dict[str, Any]] = None


def create_model(cfg: Union[Dict[str, Any], str, NLPConfig]) -> Language:
    if isinstance(cfg, str):
        cfg_dict = yaml_to_dict(cfg)
    elif isinstance(cfg, dict):
        cfg_dict = cfg
    elif isinstance(cfg, NLPConfig):
        # TODO: obviously ugly
        cfg_dict = dataclasses.asdict(cfg)
    else:
        raise ValueError(f"Expected dict, got {type(cfg)}")
    cfg_dict = resolve_alias(ALIASES, cfg_dict)
    cfg_ = dataclass_utils.into(cfg_dict, NLPConfig)
    cfg_ = correct_model_config(cfg_)
    nlp = create_lang(cfg_.lang)
    for pipe in create_pipeline(nlp, cfg_.pipeline):
        nlp.add_pipe(pipe)
    if cfg_.name:
        nlp._meta["name"] = cfg.name  # type: ignore
    nlp._meta["config"] = dataclasses.asdict(cfg_.lang)  # type: ignore
    return nlp


def create_lang(cfg: LangConfig) -> Language:
    kwargs = cfg.kwargs or {}
    if cfg.torch:
        kwargs["meta"] = merge(kwargs.get("meta", {}), {"lang": cfg.name})  # type: ignore
        if cfg.optimizer is None:
            raise ValueError("torch requires `optimizer` in configuration")
        return TorchLanguage(True, optimizer_config=cfg.optimizer, **kwargs)
    return spacy.blank(cfg.name, **kwargs)


def create_pipeline(nlp: Language, cfg: Dict[str, Any]) -> List[Pipe]:
    pipes: List[Pipe] = []
    for name, pipe_config in cfg.items():
        pipes.append(nlp.create_pipe(name, config=pipe_config))
    return pipes


_ConfigParser = Callable[[NLPConfig], NLPConfig]


def correct_model_config(cfg: NLPConfig) -> NLPConfig:
    """Parse config, Complement missing informations, resolve aliases, etc."""
    funs: List[_ConfigParser] = [
        _add_pipes,
        _add_required_pipes,
        _align_pipeline,
        _correct_trf_pipeline,
        _resolve_label,
    ]
    for f in funs:
        cfg = f(cfg)
    return cfg


# Alias definition.
# For example, `pretrained` key is converted to `pipeline.transformers_model.trf_name_or_path`.
# The aliases in config are resolved by `resolved_alias`
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
        if not cfg.labels:
            raise ValueError("`cfg.labels` required")
        cfg.pipeline = cfg.pipeline or dict()
        assert cfg.task  # for type checker
        pipe = TASK2PIPE[cfg.task]
        prev = cfg.pipeline.get(pipe, dict()) or cast(Dict[str, Any], dict())
        prev["labels"] = cfg.labels  # todo: avoid hardcoding
        cfg.pipeline[pipe] = prev
    else:
        if cfg.labels:
            raise ValueError(f"One of {TASKS} pipeline is required.")
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


_TRF_NAME_OR_PATH = "trf_name_or_path"


def _complement_trf_name(cfg: NLPConfig) -> NLPConfig:
    """For transformers pipeline.

    All transformers pipeline requires same `trf_name_or_path`, but it is a pain to write down in every pipes.
    So if there is any transformers pipeline with `trf_name_or_path`, copy it into the other pipes.
    """
    if not set(cfg.pipeline.keys()) & set(TRF_PIPES):
        return cfg

    val = _get_trf_name(cfg)
    for k, v in cfg.pipeline.items():
        if k in TRF_PIPES:
            if v:
                v[_TRF_NAME_OR_PATH] = val
            else:
                cfg.pipeline[k] = {_TRF_NAME_OR_PATH: val}
    return cfg


def _get_trf_name(cfg: NLPConfig) -> str:
    val = ""
    for _, v in cfg.pipeline.items():
        if v:
            nval = v.get(_TRF_NAME_OR_PATH, "")
            if not isinstance(nval, str):
                raise ValueError(
                    f"Value of '{_TRF_NAME_OR_PATH}' must be string, but got {nval}"
                )
            val = nval or val
    if not val:
        raise ValueError(
            f"Invalid configuration. At least one of transformer's pipe needs `{_TRF_NAME_OR_PATH}`, but the configuration is:\n{cfg}"
        )
    return val


def _correct_torch(cfg: NLPConfig) -> NLPConfig:
    """If there is any transformers pipeline, set `torch` to True"""
    if set(cfg.pipeline) & set(TRF_PIPES):
        cfg.lang.torch = True
    return cfg


def _resolve_label(cfg: NLPConfig) -> NLPConfig:
    """Resolver for ner and sequence classification label config."""
    ner = cfg.pipeline.get(TRANSFORMERS_NER)
    if ner:
        ner[LABELS] = get_ner_labels(ner[LABELS])
    seq = cfg.pipeline.get(TRANSFORMERS_SEQ_CLASSIFIER)
    if seq:
        seq[LABELS] = get_labels(seq[LABELS])
    multiseq = cfg.pipeline.get(TRANSFORMERS_MULTILABEL_SEQ_CLASSIFIER)
    if multiseq:
        multiseq[LABELS] = get_labels(multiseq[LABELS])
    return cfg
