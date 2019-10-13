"""The module trf_mixin defindes Language mixin for pytorch transformers."""
from typing import Optional, Type

from torch.optim.optimizer import Optimizer
from bedoner.lang import torch_mixin
from bedoner.torch_utils import OptimizerParameters
from transformers import AdamW, WarmupLinearSchedule
from spacy.tokens import Doc
from spacy_transformers.language import TransformersLanguage
from spacy_transformers.util import ATTRS


class TransformersLanguageMixin(torch_mixin.TorchLanguageMixin):
    """Language mixin for transformers.

    All trf components can be used with this Mixin.

    Examples:
        >>> class FooLang(TransformersLanguageMixin, juman.Japanese): # The order of inheritance is very important to properly override methods. Otherwise it will not work.
        >>>     pass
        >>> nlp = FooLang(Vocab())
        >>> nlp.add_pipe(trf_bert_component)
        >>> optim = nlp.resume_training()
        >>> nlp.update(docs, golds, optim)
    """

    @classmethod
    def install_extensions(cls):
        """Install some extensions.

        See https://github.com/explosion/spacy-pytorch-transformers#extension-attributes for details.
        """
        if Doc.get_extension(ATTRS.alignment) is None:
            TransformersLanguage.install_extensions()

    def make_optimizers(
        self,
        params: OptimizerParameters,
        optim_cls: Optional[Type[Optimizer]] = None,
        enable_scheduler: bool = True,
        **cfg
    ):
        """Create optimizer.

        Args:
            params: pytorch parameters. passed to optim_cls.__init__
            optim_cls: pytorch optimzer class. If none, create default optimizer (see the code).
            enable_scheduler: If true, create scheduler and stored it into torch_mixin.Optimizers

        Note:
            If you want to create your custom optimzers, you should create subclass and override this method,
            or create optimizer directly with the pipline components.
        """
        if optim_cls:
            return super().make_optimizers(params, optim_cls=optim_cls, **cfg)

        learning_rate: float = cfg.get("learning_rate", 5e-5)
        adam_epsilon: float = cfg.get("adam_epsilon", 1e-8)
        warmup_steps: float = cfg.get("warmup_steps", 0)
        t_total: int = cfg.get("t_total", 5)

        optimizer = AdamW(params, lr=learning_rate, eps=adam_epsilon)
        if not enable_scheduler:
            return torch_mixin.Optimizers(optimizer=optimizer)
        scheduler = WarmupLinearSchedule(
            optimizer, warmup_steps=warmup_steps, t_total=t_total
        )
        return torch_mixin.Optimizers(optimizer=optimizer, lr_scheduler=scheduler)


TransformersLanguageMixin.install_extensions()
