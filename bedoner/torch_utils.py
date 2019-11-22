"""The module torch_utils defines utilities for pytorch."""
import operator
from spacy.tokens import Doc
from typing import Any, Dict, Iterable, List, Optional, Union

import torch
import torch.nn as nn
from spacy.pipeline import Pipe

# the type torch.optim.Optimizer uses
OptimizerParameters = Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]]


class TorchPipe(Pipe):
    """Pipe wrapper for pytorch. This provides interface used by `TorchLanguageMixin`"""

    def __init__(self, device: torch.device = torch.device("cpu")):
        self._device = device

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
        """Return parameters to be optimized.

        This method is assumed to be overridden in the subclass.
        """
        self.require_model()
        model: nn.Module = self.model  # type cast
        return model.parameters()


class TensorWrapper:
    """Pytorch tensor Wrapper for efficient handling of part of batch tensors in spacy pipline"""

    def __init__(
        self, batch_tensor: torch.Tensor, index: int, length: Optional[int] = None
    ):
        self.batch_tensor = batch_tensor
        self.i = index
        self.length = length

    def get(self) -> torch.Tensor:
        if self.length is not None:
            return self.batch_tensor[self.i][: self.length]
        return self.batch_tensor[self.i]

    @property
    def tensor(self) -> torch.Tensor:
        return self.get()


def get_parameters_with_decay(
    model: nn.Module,
    weight_decay: Optional[float] = None,
    no_decay: Optional[Iterable[str]] = None,
) -> OptimizerParameters:
    """Create OptimizerParameters with weight decay.

    For parameters that include any element of `no_decay` in the name, decay is set to 0.

    Example:
        >>> params = get_parameters_with_decay(model, 0.1, ["batch_norm"])
        >>> optimizer = optim.SGD(params)
    """
    if weight_decay:
        if no_decay:
            return [
                {
                    "params": [
                        p
                        for n, p in model.named_parameters()
                        if not any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in model.named_parameters()
                        if any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": 0.0,
                },
            ]
        else:
            return [
                {
                    "params": [p for n, p in model.named_parameters()],
                    "weight_decay": weight_decay,
                }
            ]
    else:
        return model.parameters()


GoldCats = Dict[str, Union[bool, float]]


def goldcat_to_label(cats: GoldCats) -> str:
    assert len(cats)
    return max(cats.items(), key=operator.itemgetter(1))[0]


def goldcats_to_tensor(
    cats: Iterable[GoldCats], label2id: Dict[str, int]
) -> torch.Tensor:
    ids = [label2id[goldcat_to_label(cat)] for cat in cats]
    return torch.tensor(ids)


TORCH_LOSS = "torch_loss"


def get_loss_from_docs(docs: List[Doc]) -> torch.Tensor:
    losses = (doc.user_data.get(TORCH_LOSS) for doc in docs)
    losses = [loss for loss in losses if isinstance(loss, torch.Tensor)]
    if not losses:
        raise ValueError("loss is not set to docs.")
    losses = torch.stack(losses)
    return torch.sum(losses)


def add_loss_to_docs(docs: List[Doc], loss: torch.Tensor):
    doc = docs[0]
    if TORCH_LOSS in doc.user_data:
        doc.user_data[TORCH_LOSS] += loss
    else:
        doc.user_data[TORCH_LOSS] = loss
