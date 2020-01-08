from camphr.lang.torch_mixin import optim_creators
from camphr.torch_utils import OptimizerParameters
from spacy.tokens import Doc
from spacy_transformers.language import TransformersLanguage
from spacy_transformers.util import ATTRS
from torch.optim.optimizer import Optimizer
from transformers import AdamW

if Doc.get_extension(ATTRS.alignment) is None:
    TransformersLanguage.install_extensions()


@optim_creators.register("adamw")
def adamw(params: OptimizerParameters, **cfg) -> Optimizer:
    return AdamW(params, lr=cfg.get("lr", 5e-5), eps=cfg.get("eps", 1e-8))
