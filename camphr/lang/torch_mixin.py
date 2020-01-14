"""The module torch_mixin defindes Language mixin for pytorch."""
import itertools
import logging
from pathlib import Path
from typing import List, Sequence, Union

import catalogue
import srsly
import torch
import torch.optim as optim
from camphr.torch_utils import OptimizerParameters, TorchPipe, get_loss_from_docs
from spacy.gold import GoldParse  # pylint: disable=no-name-in-module
from spacy.language import Language
from spacy.tokens import Doc

logger = logging.getLogger(__name__)
optim_creators = catalogue.create("camphr", "torch_optim_creators")
OPTIM_CREATOR = "optim_creator"


@optim_creators.register("base")
def base(params: OptimizerParameters, **cfg) -> optim.Optimizer:
    return optim.SGD(params, lr=cfg.get("lr", 0.01))


class TorchLanguageMixin:
    """spacy.Language mixin for pytorch.

    This mixin manages all `TorchPipe` components for the sake of training.

    Examples:
        >>> class FooLang(TorchLanguageMixin, juman.Japanese): # The order of inheritance is very important to properly override methods. Otherwise it will not work.
        >>>     pass
        >>> nlp = FooLang(Vocab())
        >>> nlp.add_pipe(foo_torch_pipe)
        >>> optim = nlp.resume_training()
        >>> nlp.update(docs, golds, optim)
    """

    def resume_training(self, **kwargs) -> optim.Optimizer:
        """Gather all torch parameters in each `TorchPipe`s, and create an optimizer.

        Args:
            kwargs: passed to `self.make_optimizers`
        """
        assert hasattr(self, "pipeline")

        params = itertools.chain.from_iterable(
            pipe.optim_parameters()
            for _, pipe in self.pipeline
            if isinstance(pipe, TorchPipe)
        )
        return self.make_optimizer(params, **kwargs)

    @property
    def device(self) -> torch.device:
        if not hasattr(self, "_device"):
            self._device = torch.device("cpu")
            self.to(self._device)
        else:
            for pipe in self.get_torch_pipes():
                assert self._device.type == pipe.device.type
        return self._device

    def get_torch_pipes(self) -> List[TorchPipe]:
        return [pipe for _, pipe in self.pipeline if isinstance(pipe, TorchPipe)]

    def to(self, device: torch.device) -> bool:
        flag = False
        for pipe in self.get_torch_pipes():
            flag = True
            pipe.to(device)
        self._device = device
        return flag

    def make_optimizer(self, params: OptimizerParameters, **cfg) -> optim.Optimizer:
        """Make optimizer.

        If you want to create your custom optimzers, you should create custom `make optimzer` function and register it in `optimizer_creator`.
        """
        name = self.meta.get(OPTIM_CREATOR, "")
        if name:
            fn = optim_creators.get(name)
            if fn:
                return fn(params, **cfg)
        return optim_creators.get("base")(params, **cfg)

    def update(
        self,
        docs: Sequence[Doc],
        golds: Sequence[GoldParse],
        optimizer: optim.Optimizer,
        debug: bool = False,
    ):
        """Update `TorchPipe` models in pipline."""
        docs, golds = self._format_docs_and_golds(docs, golds)

        for _, pipe in self.pipeline:
            pipe.update(docs, golds)

        loss = get_loss_from_docs(docs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if debug:
            logger.info(f"Loss: {loss.detach().item()}")

    def to_disk(self, path: Union[str, Path], exclude=tuple(), disable=None):
        """Overrides Language.to_disk to save `nlp.lang` properly"""
        path = Path(path)
        super().to_disk(path, exclude, disable)
        meta = self.meta
        meta["lang"] = self.lang
        srsly.write_json(path / "meta.json", meta)


class TorchLanguage(TorchLanguageMixin, Language):
    """This class is useful for type annotation.

    Python doesn't have `Intersection` type, so you cannot use `TorchLanguageMixin` for type annotation well.
    So why `TorchLanguageMixin` is there? - That's because you can use the functionality of
    `TorchLanguageMixin` with already implemented Language class, for example, `spacy.English`.
    """

    def __init__(self) -> None:
        raise NotImplementedError(
            """This class is only for type annotation.
            You should use `TorchLanguageMixin` to use the functionality instea instead."""
        )
