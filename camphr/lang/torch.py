"""The module torch_mixin defindes Language mixin for pytorch."""
import itertools
import logging
from pathlib import Path
from typing import Any, Dict, List, Sequence, Union

import srsly
import torch
import torch.optim as optim
from spacy.gold import GoldParse  # pylint: disable=no-name-in-module
from spacy.language import Language
from spacy.tokens import Doc

from camphr.torch_utils import OptimizerParameters, TorchPipe, get_loss_from_docs
from camphr.utils import get_defaults, get_requirements_line, import_attr

logger = logging.getLogger(__name__)


class TorchLanguage(Language):
    """spacy.Language for pytorch.

    This calss manages all `TorchPipe` components for the sake of training.

    Examples:
        >>> nlp = TorchLanguage(Vocab(), meta={"lang": "en"})
        >>> nlp.add_pipe(foo_torch_pipe)
        >>> optim = nlp.resume_training()
        >>> nlp.update(docs, golds, optim)
    """

    LANG_FACTORY = "camphr_torch"

    def __init__(
        self,
        vocab=True,
        make_doc=True,
        max_length=10 ** 6,
        meta={},
        optimizer_config: Dict[str, Any] = {},
        **kwargs,
    ):
        meta = dict(meta)
        meta[
            "lang_factory"
        ] = self.LANG_FACTORY  # lang_factory is necessary when restoring.
        meta.setdefault("requirements", []).append(get_requirements_line())
        self.lang = meta.get("lang", "")
        self.Defaults = get_defaults(self.lang)
        self.optimizer_config = optimizer_config
        super().__init__(vocab, make_doc, max_length, meta=meta, **kwargs)

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
        return self.create_optimizer(params, **kwargs)

    def require_optimizer_config(self):
        assert isinstance(
            self.optimizer_config, dict
        ), f"`self.optimizer_config` must be set."

    def create_optimizer(
        self: Language, params: OptimizerParameters, **kwargs
    ) -> optim.Optimizer:
        cls = import_attr(self.optimizer_config["class"])
        return cls(params, **self.optimizer_config.get("params", {}))

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

    def to_disk(self, path: Union[str, Path], exclude=tuple(), disable=None):
        """Overrides Language.to_disk to save `nlp.lang` properly."""
        path = Path(path)
        super().to_disk(path, exclude, disable)
        meta = self.meta
        meta["lang"] = self.lang
        srsly.write_json(path / "meta.json", meta)


def get_torch_nlp(lang: str, **cfg):
    return TorchLanguage(meta={"lang": lang}, **cfg)
