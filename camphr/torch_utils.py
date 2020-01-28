"""The module torch_utils defines utilities for pytorch."""
import contextlib
import operator
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Union, cast

import torch
import torch.nn as nn
from spacy.pipeline import Pipe
from spacy.tokens import Doc
from torch._C import is_grad_enabled  # type: ignore

# the type torch.optim.Optimizer uses
OptimizerParameters = Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]]


class TorchPipe(Pipe):
    """Pipe wrapper for pytorch. This provides interface used by `TorchLanguageMixin`"""

    def __init__(self, vocab, model=True, **cfg):
        self.vocab = vocab
        self.model = model
        self._device = torch.device("cpu")
        self.cfg = cfg

    @property
    def device(self):
        if not hasattr(self, "_device"):
            self._device = torch.device("cpu")
        return self._device

    def to(self, device: torch.device):
        self._device = device
        if self.model and not isinstance(self.model, bool):
            self.model.to(device)

    def optim_parameters(self) -> OptimizerParameters:
        """Return parameters to be optimized."""
        self.require_model()
        if self.cfg.get("freeze"):
            return []
        return cast(nn.Module, self.model).parameters()


@dataclass
class TensorWrapper:
    """Pytorch tensor Wrapper for efficient handling of part of batch tensors in spacy pipline"""

    batch_tensor: torch.Tensor
    i: int
    length: Optional[int] = None

    def get(self) -> torch.Tensor:
        if self.length is not None:
            return self.batch_tensor[self.i, : self.length]
        return self.batch_tensor[self.i]


GoldCat = Dict[str, Union[bool, float]]


def goldcat_to_label(goldcat: GoldCat) -> str:
    assert len(goldcat)
    return max(goldcat.items(), key=operator.itemgetter(1))[0]


def goldcats_to_tensor(
    goldcats: Iterable[GoldCat], label2id: Dict[str, int]
) -> torch.Tensor:
    ids = [label2id[goldcat_to_label(cat)] for cat in goldcats]
    return torch.tensor(ids)


TORCH_LOSS = "torch_loss"


def get_loss_from_docs(docs: Iterable[Doc]) -> torch.Tensor:
    _losses = (doc.user_data.get(TORCH_LOSS) for doc in docs)
    losses = [loss for loss in _losses if isinstance(loss, torch.Tensor)]
    if not losses:
        raise ValueError("loss is not set to docs.")
    tlosses = torch.stack(losses)
    return torch.sum(tlosses)


def add_loss_to_docs(docs: List[Doc], loss: torch.Tensor):
    """Add loss to docs' existing loss. """
    doc = docs[0]
    if TORCH_LOSS in doc.user_data:
        doc.user_data[TORCH_LOSS] += loss
    else:
        doc.user_data[TORCH_LOSS] = loss


@contextlib.contextmanager
def set_grad(grad: bool) -> Iterator[None]:
    prev = is_grad_enabled()
    torch.set_grad_enabled(grad)
    yield
    torch.set_grad_enabled(prev)
