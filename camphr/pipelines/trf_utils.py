import dataclasses
import pickle
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Sized, Type, cast

import torch
import torch.nn as nn
import transformers
from camphr.lang.torch_mixin import optim_creators
from camphr.torch_utils import (
    OptimizerParameters,
    TensorWrapper,
    get_parameters_with_decay,
)
from spacy.tokens import Doc
from spacy.vocab import Vocab
from tokenizations import get_alignments
from torch.optim.optimizer import Optimizer
from transformers import AdamW
from typing_extensions import Protocol

from .trf_auto import get_trf_config_cls, get_trf_name


@optim_creators.register("adamw")
def adamw(params: OptimizerParameters, **cfg) -> Optimizer:
    return AdamW(params, lr=cfg.get("lr", 5e-5), eps=cfg.get("eps", 1e-8))


class ATTRS:
    tokens = "transformers_tokens"
    token_ids = "transformers_token_ids"
    cleaned_tokens = "transformers_cleaned_tokens"
    special_tokens = "transformers_special_tokens"
    align = "transformers_align"
    last_hidden_state = "transformers_last_hidden_state"
    batch_inputs = "transformers_batch_inputs"


def _get_transformers_align(doc):
    trf_tokens = doc._.get(ATTRS.cleaned_tokens)
    return get_alignments([token.text for token in doc], trf_tokens)[0]


for attr in [
    ATTRS.tokens,
    ATTRS.token_ids,
    ATTRS.cleaned_tokens,
    ATTRS.last_hidden_state,
    ATTRS.batch_inputs,
]:
    Doc.set_extension(attr, default=None)
Doc.set_extension(ATTRS.align, getter=_get_transformers_align)


TRF_CONFIG = "trf_config"
VOCAB_SIZE = "vocab_size"
TRF_NAME = "trf_name"
CONVERT_LABEL = "convert_label"


def get_last_hidden_state_from_docs(docs: Iterable[Doc]) -> torch.Tensor:
    # assumed that the batch tensor of all docs is stored in the extension.
    x: TensorWrapper = next(iter(docs))._.get(ATTRS.last_hidden_state)
    return x.batch_tensor


def get_dropout(config: transformers.PretrainedConfig) -> float:
    if isinstance(config, transformers.BertConfig):
        return config.hidden_dropout_prob
    if hasattr(config, "dropout"):
        return config.dropout
    return 0.1


def _setdefault(obj: Any, k: str, v: Any) -> Any:
    if hasattr(obj, k):
        return getattr(obj, k)
    setattr(obj, k, v)
    return v


def _setdefaults(obj: Any, kv: Dict[str, Any]):
    for k, v in kv.items():
        _setdefault(obj, k, v)


_SUMMARY_DEFAULTS = {"summary_activation": "tanh"}
_SUMMARY_TYPE = "summary_type"
_LAST_DROPOUT = "summary_last_dropout"


def _set_default_summary_type(config: transformers.PretrainedConfig):
    if isinstance(config, transformers.BertConfig):
        _setdefault(config, _SUMMARY_TYPE, "first")
    elif isinstance(config, transformers.XLNetConfig):
        _setdefault(config, _SUMMARY_TYPE, "last")


def set_default_config_for_sequence_summary(config: transformers.PretrainedConfig):
    """Set default value for transformers.SequenceSummary"""
    _setdefaults(config, _SUMMARY_DEFAULTS)
    _set_default_summary_type(config)
    _setdefault(config, _LAST_DROPOUT, get_dropout(config))


class TrfModelForTaskBase(nn.Module):
    def __init__(self, config: transformers.PretrainedConfig):
        super().__init__()
        self.config = config


class TrfOptimMixin:
    def optim_parameters(self) -> OptimizerParameters:
        no_decay = self.cfg.get("no_decay")
        weight_decay = self.cfg.get("weight_decay")
        return get_parameters_with_decay(self.model, no_decay, weight_decay)


class _TrfSavePathGetter:
    def _trf_path(self, path: Path) -> Path:
        return path / self.cfg[TRF_NAME]


class SerializationMixinForTrfTask(_TrfSavePathGetter):
    _MODEL_PTH = "model.pth"
    _CFG_PKL = "cfg.pkl"
    model_cls: Type[TrfModelForTaskBase] = TrfModelForTaskBase

    def to_disk(self, path: Path, exclude=tuple(), **kwargs):
        path.mkdir(exist_ok=True)
        model: TrfModelForTaskBase = self.model
        config_save_path = self._trf_path(path)
        config_save_path.mkdir(exist_ok=True)
        model.config.save_pretrained(str(config_save_path))
        torch.save(model.state_dict(), str(path / self._MODEL_PTH))

        with (path / self._CFG_PKL).open("wb") as f:
            cfg = {k: v for k, v in self.cfg.items() if k not in exclude}
            pickle.dump(cfg, f)

    def from_disk(self, path: Path, exclude=tuple(), **kwargs):
        with (path / self._CFG_PKL).open("rb") as f:
            self.cfg = pickle.load(f)

        config = get_trf_config_cls(self.cfg[TRF_NAME]).from_pretrained(
            str(self._trf_path(path))
        )
        model = self.model_cls(config)
        model.load_state_dict(
            torch.load(str(path / self._MODEL_PTH), map_location=self.device)
        )
        model.eval()
        self.model = model

        return self


class TrfAutoProtocol(Protocol):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        ...


class TrfAutoMixin(_TrfSavePathGetter):

    _MODEL_PTH = "model.pth"
    _CFG_PKL = "cfg.pkl"
    _MODEL_CLS_GETTER: Callable[[str], Type[TrfAutoProtocol]]

    @classmethod
    def Model(cls, trf_name_or_path: str, **cfg):
        return cls._MODEL_CLS_GETTER(trf_name_or_path).from_pretrained(
            trf_name_or_path, **cfg
        )

    @classmethod
    def from_pretrained(cls, vocab: Vocab, name_or_path: str, **cfg):
        """Load pretrained model."""
        name = get_trf_name(name_or_path)
        return cls(vocab, model=cls.Model(name_or_path), trf_name=name, **cfg)

    def to_disk(self, path: Path, exclude=tuple(), **kwargs):
        path.mkdir(exist_ok=True)
        model_path = self._trf_path(path)
        model_path.mkdir(exist_ok=True)
        self.model.save_pretrained(str(model_path))
        with (path / self._CFG_PKL).open("wb") as f:
            pickle.dump(self.cfg, f)

    def from_disk(self, path: Path, exclude=tuple(), **kwargs):
        with (path / self._CFG_PKL).open("rb") as f:
            self.cfg = pickle.load(f)
        self.model = self.Model(str(self._trf_path(path)))
        return self


@dataclasses.dataclass
class TransformersInput:
    input_ids: torch.Tensor
    token_type_ids: torch.Tensor
    attention_mask: torch.Tensor
    input_len: torch.LongTensor

    def __iter__(self) -> Iterator["TransformersInput"]:
        for i in range(len(cast(Sized, self.input_ids))):
            row = {k: getattr(self, k)[i] for k in self.tensor_field_names}
            row = TransformersInput(**row)
            yield row

    def to(self, *args, **kwargs):
        for k in self.tensor_field_names:
            getattr(self, k).to(*args, **kwargs)

    @property
    def tensor_field_names(self) -> List[str]:
        return [
            field.name
            for field in dataclasses.fields(self)
            if field.type is torch.Tensor or field.type is torch.LongTensor
        ]

    @property
    def model_input(self):
        output = {k: getattr(self, k) for k in self.tensor_field_names}
        del output["input_len"]
        return output
