import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Union

import torch
import torch.nn as nn
import transformers as trf
from spacy.gold import GoldParse
from spacy.tokens import Doc
from spacy.vocab import Vocab
from spacy_transformers.util import ATTRS

from bedoner.torch_utils import (
    OptimizerParameters,
    TensorWrapper,
    TorchPipe,
    get_parameters_with_decay,
)

TrfConfig = Union[trf.BertConfig, trf.XLNetConfig]

TRF_CONFIG = "trf_config"
VOCAB_SIZE = "vocab_size"
TRF_NAME = "trf_name"
CONVERT_LABEL = "convert_label"


class Errors:
    E1 = "{} is not supported"


def get_last_hidden_state_from_docs(docs: Iterable[Doc]) -> torch.Tensor:
    # assumed that the batch tensor of all docs is stored in the extension.
    x: TensorWrapper = next(iter(docs))._.get(ATTRS.last_hidden_state)
    return x.batch_tensor


def get_dropout(config: TrfConfig) -> float:
    if isinstance(config, trf.BertConfig):
        return config.hidden_dropout_prob
    elif isinstance(config, trf.XLNetConfig):
        return config.dropout
    raise ValueError(Errors.E1.format(type(config)))


def _setdefault(obj: Any, k: str, v: Any) -> Any:
    if hasattr(obj, k):
        return getattr(obj, k)
    setattr(obj, k, v)
    return v


def _setdefaults(obj: Any, kv: Dict[str, Any]):
    for k, v in kv.items():
        _setdefault(obj, k, v)


# TODO: https://github.com/huggingface/transformers/issues/1845
_SUMMARY_DEFAULTS = {"summary_use_proj": False, "summary_activation": "tanh"}


_SUMMARY_TYPE = "summary_type"
_LAST_DROPOUT = "summary_last_dropout"


def _set_default_summary_type(config: TrfConfig):
    if isinstance(config, trf.BertConfig):
        _setdefault(config, _SUMMARY_TYPE, "first")
    elif isinstance(config, trf.XLNetConfig):
        _setdefault(config, _SUMMARY_TYPE, "last")


def set_default_config_for_sequence_summary(config: TrfConfig):
    """Set default value for transformers.SequenceSummary"""
    _setdefaults(config, _SUMMARY_DEFAULTS)
    _set_default_summary_type(config)
    _setdefault(config, _LAST_DROPOUT, get_dropout(config))


class TrfModelForTaskBase(nn.Module):
    def __init__(self, config: TrfConfig):
        super().__init__()
        self.config = config


class TrfPipeForTaskBase(TorchPipe):
    """Base class for end task"""

    _MODEL_PTH = "model.pth"
    _CFG_PKL = "cfg.pkl"

    def __init__(self, vocab, model=True, **cfg):
        self.vocab = vocab
        self.model = model
        self.cfg = cfg

    @classmethod
    def from_pretrained(cls, vocab: Vocab, name_or_path: str, **cfg):
        cfg[TRF_NAME] = name_or_path
        model = cls.Model(from_pretrained=True, **cfg)
        cfg[TRF_CONFIG] = dict(model.config.to_dict())
        return cls(vocab, model=model, **cfg)

    @classmethod
    def Model(cls, **cfg) -> TrfModelForTaskBase:
        cfg.setdefault(TRF_CONFIG, {})
        if cfg.get("from_pretrained"):
            config = cls.trf_config_cls.from_pretrained(cfg[TRF_NAME])
            for k, v in cfg[TRF_CONFIG].items():
                setattr(config, k, v)
            model = cls.model_cls(config)
        else:
            if VOCAB_SIZE in cfg[TRF_CONFIG]:
                vocab_size = cfg[TRF_CONFIG][VOCAB_SIZE]
                cfg[TRF_CONFIG]["vocab_size_or_config_json_file"] = vocab_size
            model = cls.model_cls(cls.trf_config_cls.from_dict(cfg[TRF_CONFIG]))
        return model

    def predict(self, docs: Iterable[Doc]):
        raise NotImplementedError

    def set_annotations(self, docs: Iterable[Doc], preds: Any):
        raise NotImplementedError

    def update(self, docs: List[Doc], golds: Iterable[GoldParse]):
        raise NotImplementedError

    def optim_parameters(self) -> OptimizerParameters:
        no_decay = self.cfg.get("no_decay")
        weight_decay = self.cfg.get("weight_decay")
        return get_parameters_with_decay(self.model, no_decay, weight_decay)

    def to_disk(self, path: Path, exclude=tuple(), **kwargs):
        path.mkdir(exist_ok=True)
        model: TrfModelForTaskBase = self.model
        model.config.save_pretrained(path)
        torch.save(model.state_dict(), str(path / self._MODEL_PTH))

        with (path / self._CFG_PKL).open("wb") as f:
            cfg = {k: v for k, v in self.cfg.items() if k not in exclude}
            pickle.dump(cfg, f)

    def from_disk(self, path: Path, exclude=tuple(), **kwargs):
        with (path / self._CFG_PKL).open("rb") as f:
            self.cfg = pickle.load(f)
        config = self.trf_config_cls.from_pretrained(path)
        model = self.model_cls(config)
        model.load_state_dict(
            torch.load(str(path / self._MODEL_PTH), map_location=self.device)
        )
        model.eval()
        self.model = model

        return self
