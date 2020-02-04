"""Defines utility functions, classes, mixins for transformers pipelines."""
from contextlib import contextmanager
import dataclasses
import pickle
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    Iterator,
    List,
    Sized,
    Type,
    TypeVar,
    cast,
)

import torch
import torch.nn as nn
import transformers
from spacy.language import Language
from spacy.tokens import Doc
from spacy.vocab import Vocab
from tokenizations import get_alignments
from typing_extensions import Protocol, Literal
from spacy.gold import GoldParse

from camphr.pipelines.utils import UserHooksMixin
<<<<<<< HEAD
from camphr.torch_utils import TensorWrapper, TorchPipe, set_grad
=======
from camphr.torch_utils import TensorWrapper
>>>>>>> origin/master

from .auto import get_trf_config_cls, get_trf_name


class ATTRS:
    """Attribute names for spacy.Underscore"""

    tokens = "transformers_tokens"
    token_ids = "transformers_token_ids"
    cleaned_tokens = "transformers_cleaned_tokens"
    special_tokens = "transformers_special_tokens"
    align = "transformers_align"
    last_hidden_state = "transformers_last_hidden_state"
    batch_inputs = "transformers_batch_inputs"


def _get_transformers_align(doc: Doc) -> List[List[int]]:
    """Get tokens alignment from spacy tokens to transformers tokens"""
    trf_tokens = doc._.get(ATTRS.cleaned_tokens)
    return get_alignments([token.text for token in doc], trf_tokens)[0]


def _set_extensions():
    for attr in [
        ATTRS.tokens,
        ATTRS.token_ids,
        ATTRS.cleaned_tokens,
        ATTRS.last_hidden_state,
        ATTRS.batch_inputs,
    ]:
        Doc.set_extension(attr, default=None)
    Doc.set_extension(ATTRS.align, getter=_get_transformers_align)


_set_extensions()


TRF_CONFIG = "trf_config"
VOCAB_SIZE = "vocab_size"
TRF_NAME = "trf_name"
CONVERT_LABEL = "convert_label"
LABEL_WEIGHTS = "label_weights"
LABELS = "labels"


def get_last_hidden_state_from_docs(docs: Iterable[Doc]) -> torch.Tensor:
    """Get transformers text embedding from docs.
    
    Useful for downstream task pipelines.
    """
    # assumed that the batch tensor of all docs is stored in the extension.
    x: TensorWrapper = next(iter(docs))._.get(ATTRS.last_hidden_state)
    return x.batch_tensor


def get_dropout(config: transformers.PretrainedConfig) -> float:
    """Get dropout rate from config"""
    if isinstance(config, transformers.BertConfig):
        return config.hidden_dropout_prob
    if hasattr(config, "dropout"):
        return config.dropout
    return 0.1


class TrfModelForTaskBase(nn.Module):
    def __init__(self, config: transformers.PretrainedConfig):
        super().__init__()
        self.config = config


class _TrfSavePathGetter:
    def _trf_path(self, path: Path) -> Path:
        return path / self.cfg[TRF_NAME]  # type: ignore


class SerializationMixinForTrfTask(_TrfSavePathGetter):
    """Mixin for transoformers pipeline

    Constraints: `TorchPipe`
    """

    _MODEL_PTH = "model.pth"
    _CFG_PKL = "cfg.pkl"
    model_cls: Type[TrfModelForTaskBase]

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
            torch.load(
                str(path / self._MODEL_PTH), map_location=self.device  # type: ignore
            )
        )
        model.eval()
        self.model = model
        return self


class FromNLPMixinForTrfTask:
    """Mixin for transformers task pipeline.

    Constraints: `TorchPipe`
    """

    @classmethod
    def from_pretrained(cls, vocab: Vocab, trf_name_or_path: str, **cfg):
        """Load pretrained model."""
        name = get_trf_name(trf_name_or_path)
        return cls(  # type: ignore
            vocab,
            model=cls.Model(trf_name_or_path, **cfg),  # type: ignore
            trf_name=name,
            **cfg
        )

    @classmethod
    def from_nlp(cls, nlp: Language, **cfg):
        if cfg.get("trf_name_or_path"):
            return cls.from_pretrained(nlp.vocab, **cfg)
        return cls(nlp.vocab)  # type: ignore


class PipeProtocol(Protocol):
    """For mypy. See https://mypy.readthedocs.io/en/latest/more_types.html#mixin-classes"""

    cfg: Dict[str, Any]


class LabelsMixin:
    """Mixin for pipes which has labels.


    Constraints: `Pipe`
    Optional constraints:
        - `UserHooksMixin`: to use `def convert_label`
    """

    @property
    def labels(self: PipeProtocol) -> List[str]:
        # The reason to return empty list is that `spacy.Language` calls `Pipe.labels` when restoring,
        # before the `labels` is not set to this Pipe.
        # For the same reason cache cannot be used.
        return self.cfg.get(LABELS, [])

    @property
    def label2id(self: PipeProtocol) -> Dict[str, int]:
        # `lru_cache` is not adopted because mypy doesn't support `Decorated property`
        # https://github.com/python/mypy/issues/1362
        if not hasattr(self, "_label2id"):
            self._label2id = {v: i for i, v in enumerate(self.labels)}  # type: ignore
        return self._label2id  # type: ignore

    @property
    def label_weights(self: PipeProtocol) -> torch.Tensor:
        if not hasattr(self, "_label_weights"):
            weights_map = self.cfg.get(LABEL_WEIGHTS)
            weights = torch.ones(len(self.label2id))  # type: ignore
            if weights_map:
                assert len(weights_map) == len(self.label2id)  # type: ignore
                for k, v in weights_map.items():
                    weights[self.label2id[k]] = v  # type: ignore
            self._label_weights = weights
        return self._label_weights  # type: ignore

    def convert_label(self: UserHooksMixin, label: str) -> str:  # type:ignore
        fn = self.user_hooks.get(CONVERT_LABEL)
        if fn:
            return fn(label)
        return label


class TrfObj(Protocol):
    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path, *inputs, **kwargs
    ) -> "TrfObj":
        ...

    def save_pretrained(self, *args, **kwargs):
        ...


T = TypeVar("T", bound=TrfObj)


class TrfAutoMixin(_TrfSavePathGetter, Generic[T]):
    """Mixin for transformers' `AutoModel` and `AutoTokenizer`

    Required to be combined with `TorchPipe`
    """

    _MODEL_PTH = "model.pth"
    _CFG_PKL = "cfg.pkl"
    _MODEL_CLS_GETTER: Callable[[str], Type[T]]
    model: T

    @classmethod
    def Model(cls, trf_name_or_path: str, **cfg):
        _cls = cls._MODEL_CLS_GETTER(trf_name_or_path)
        return _cls.from_pretrained(trf_name_or_path, **cfg)

    @classmethod
    def from_pretrained(cls, vocab: Vocab, trf_name_or_path: str, **cfg):
        """Load pretrained model."""
        name = get_trf_name(trf_name_or_path)
        model = cls(  # type: ignore
            vocab, model=cls.Model(trf_name_or_path), trf_name=name, **cfg
        )
        return model

    @classmethod
    def from_nlp(cls, nlp: Language, **cfg):
        if cfg.get("trf_name_or_path"):
            return cls.from_pretrained(nlp.vocab, **cfg)
        return cls(nlp.vocab)  # type: ignore

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


class ComputeLossMixin:
    def update(self, docs: List[Doc], golds: List[GoldParse]):
        return self.compute_loss(docs, golds, "train")

    def eval(self, docs: List[Doc], golds: List[GoldParse]):
        return self.compute_loss(docs, golds, "eval")

    def compute_loss(
        self: TorchPipe,
        docs: List[Doc],
        golds: List[GoldParse],
        mode: Literal["train", "eval"],
    ):
        raise NotImplementedError

    @contextmanager
    def switch(self: TorchPipe, mode: Literal["train", "eval"]):
        self.require_model()
        if mode == "train":
            self.model.train()
            grad = True
        else:
            self.model.eval()
            grad = False
        with set_grad(grad):
            yield


@dataclasses.dataclass
class TransformersInput:
    input_ids: torch.Tensor
    token_type_ids: torch.Tensor
    attention_mask: torch.Tensor
    input_len: torch.LongTensor

    def __iter__(self) -> Iterator["TransformersInput"]:
        for i in range(len(cast(Sized, self.input_ids))):
            _row = {k: getattr(self, k)[i] for k in self.tensor_field_names}
            row = TransformersInput(**_row)
            yield row

    def to(self, *args, **kwargs):
        for k in self.tensor_field_names:
            t = getattr(self, k).to(*args, **kwargs)
            setattr(self, k, t)

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
