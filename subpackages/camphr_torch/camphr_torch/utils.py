"""Defines utilities for pytorch."""
import contextlib
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, Optional, Sequence, Union, cast

from spacy.pipeline import Pipe
from spacy.tokens import Doc
import torch
from torch._C import is_grad_enabled
import torch.nn as nn

from camphr_core.utils import GoldCat, goldcat_to_label

# the type torch.optim.Optimizer uses
OptimizerParameters = Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]]


class TorchPipe(Pipe):
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
    """Pytorch tensor wrapper for efficient handling of part of batch tensors in spacy pipeline"""

    batch_tensor: torch.Tensor
    i: int
    length: Optional[int] = None

    def get(self) -> torch.Tensor:
        if self.length is not None:
            return self.batch_tensor[self.i, : self.length]
        return self.batch_tensor[self.i]


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


def add_loss_to_docs(docs: Sequence[Doc], loss: torch.Tensor):
    """Add loss to docs' existing loss."""
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


def beamsearch(probs: torch.Tensor, k: int) -> torch.Tensor:
    """Beam search for sequential probabilities.

    Args:
        probs: tensor of shape (length, d). requires d > 0. Assumed all items in `probs` in range [0, 1].
        k: beam width
    Returns: (k, length) tensor
    """
    assert len(probs.shape) == 2
    if len(probs) == 0:
        return torch.zeros(k, 0)
    if k == 1:
        return probs.argmax(-1)[None, :]

    # We calculate top k-th argmax of E = p0p1p2..pn-1.
    # To avoid over(under)flow, evaluete log(E) instead of E.
    # By doing this, E can be evaluated by addition.
    data = probs.to(torch.double).log()
    _, m = data.shape
    scores, candidates = torch.topk(data[0], k=min(k, m))
    candidates = candidates[:, None]
    for row in data[1:]:
        z = (scores[:, None] + row[None, :]).flatten()
        scores, flat_idx = torch.topk(z, k=min(k, len(z)))
        i, j = flat_idx // m, flat_idx % m
        candidates = torch.cat([candidates[i], j[:, None]], dim=-1)
    return candidates
