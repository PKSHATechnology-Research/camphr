"""The module torch_mixin defindes Language mixin for pytorch."""
from bedoner.pipelines.trf_model import BertModel
import itertools
from typing import List, Optional, Type, cast

import torch
import torch.optim as optim
from bedoner.torch_utils import OptimizerParameters, TorchPipe
from dataclasses import dataclass
from spacy.errors import Errors as SpacyErrors
from spacy.gold import GoldParse  # pylint: disable=no-name-in-module
from spacy.tokens import Doc


@dataclass
class Optimizers:
    """Container for optimizer and scheduler."""

    optimizer: optim.Optimizer
    lr_scheduler: Optional[optim.lr_scheduler._LRScheduler] = None

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()
        if self.lr_scheduler:
            # TODO: remove cast once bug in pytorch stub file is fixed (https://github.com/pytorch/pytorch/pull/26531).
            self.lr_scheduler.step(cast(int, None))


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
            Doc.set_extension("loss", default=None)

    def resume_training(self, **kwargs) -> Optimizers:
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
        return self.make_optimizers(params, **kwargs)

    def to(self, device: torch.device) -> bool:
        flag = False
        for _, pipe in self.pipeline:
            if isinstance(pipe, TorchPipe):
                print(pipe)
                flag = True
                pipe.to(device)
        return flag

    def make_optimizers(
        self,
        params: OptimizerParameters,
        optim_cls: Optional[Type[optim.Optimizer]] = None,
        **cfg
    ) -> Optimizers:
        """Make optimizer and scheduler (if necessary), wrap them in `Optimizers` and returns it.

        If you want to create your custom optimzers, you should create subclass and override this method,
        or create optimizer directly with the pipline components.
        """
        if optim_cls:
            return Optimizers(optim_cls(params, **cfg))

        return Optimizers(optim.SGD(params, lr=0.01))

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
        optimizers: Optimizers,
        debug: bool = False,
    ):
        """Update `TorchPipe` models in pipline."""
        docs, golds = self._format_docs_and_golds(docs, golds)

        for _, pipe in self.pipeline:
            pipe.update(docs, golds)

        loss = torch.mean(torch.stack([doc._.loss for doc in docs]))
        optimizers.zero_grad()
        loss.backward()
        optimizers.step()
        if debug:
            print(loss)


TorchLanguageMixin.install_extensions()
