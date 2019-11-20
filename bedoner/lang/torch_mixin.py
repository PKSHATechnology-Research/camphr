"""The module torch_mixin defindes Language mixin for pytorch."""
import itertools
import logging
from typing import List

import torch
import torch.optim as optim
from spacy.errors import Errors as SpacyErrors
from spacy.gold import GoldParse  # pylint: disable=no-name-in-module
from spacy.tokens import Doc
import catalogue

from bedoner.torch_utils import OptimizerParameters, TorchPipe

logger = logging.getLogger(__name__)
optim_creators = catalogue.create("bedoner", "torch_optim_creators")
OPTIM_CREATOR = "optim_creator"


@optim_creators.register("base")
def base(params: OptimizerParameters, **cfg) -> optim.Optimizer:
    return optim.SGD(params, lr=cfg.get("lr", 0.01))


class TorchLanguageMixin:
    """Language mixin for pytorch.

    This mixin manages all `TorchPipe` components for the sake of training.

    Notes:
        `spacy.Doc._.loss` is set. All `TorchPipe` should add its loss into it when training.
        This mixin backwards it, i.e. call `spacy.Doc._.loss.backward()` when `nlp.update` is called.

    Examples:
        >>> class FooLang(TorchLanguageMixin, juman.Japanese): # The order of inheritance is very important to properly override methods. Otherwise it will not work.
        >>>     pass
        >>> nlp = FooLang(Vocab())
        >>> nlp.add_pipe(foo_torch_pipe)
        >>> optim = nlp.resume_training()
        >>> nlp.update(docs, golds, optim)
    """

    @classmethod
    def install_extensions(cls):
        """Add some extensions.

        - loss (torch.Tensor): holds loss, used when `nlp.udpate`

        Refs:
            https://spacy.io/usage/processing-pipelines#custom-components-attributes
        """
        if Doc.get_extension("loss") is None:
            Doc.set_extension("loss", default=0)

    def resume_training(self, **kwargs) -> optim.Optimizer:
        """Gather torch parameters in `TorchPipe`, and create optimizers.

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

    def _format_docs_and_golds(self, docs, golds):
        # TODO: remove this method after PR merged (https://github.com/explosion/spaCy/pull/4316)
        expected_keys = ("words", "tags", "heads", "deps", "entities", "cats", "links")
        gold_objs = []
        doc_objs = []
        for doc, gold in zip(docs, golds):
            if isinstance(doc, str):
                doc = self.make_doc(doc)
            if not isinstance(gold, GoldParse):
                unexpected = [k for k in gold if k not in expected_keys]
                if unexpected:
                    err = SpacyErrors.E151.format(unexp=unexpected, exp=expected_keys)
                    raise ValueError(err)
                gold = GoldParse(doc, **gold)
            doc_objs.append(doc)
            gold_objs.append(gold)

        return doc_objs, gold_objs

    def update(
        self,
        docs: List[Doc],
        golds: List[GoldParse],
        optimizer: optim.Optimizer,
        debug: bool = False,
    ):
        """Update `TorchPipe` models in pipline."""
        docs, golds = self._format_docs_and_golds(docs, golds)

        for _, pipe in self.pipeline:
            pipe.update(docs, golds)

        loss = torch.mean(torch.stack([doc._.loss for doc in docs]))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if debug:
            logger.info(f"Loss: {loss.detach().item()}")


TorchLanguageMixin.install_extensions()
