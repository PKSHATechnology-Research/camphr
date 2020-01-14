import re
import sys

from camphr.lang.torch_mixin import TorchLanguageMixin
from spacy.util import get_lang_class, set_lang_class

TORCH = "Torch"


def _is_torchlang_cls_name(name: str) -> bool:
    return name.endswith(TORCH)


def _get_lang_name(name: str) -> str:
    return re.sub(re.escape(TORCH), "", name).lower()


def get_torch_lang_cls(name: str):
    return getattr(sys.modules[__name__], name.title() + TORCH)


__all__ = ["get_torch_lang_cls"]


def __getattr__(name: str):
    if name == "__path__":
        return
    if not _is_torchlang_cls_name(name):
        raise ImportError(f"cannot import name '{name}' from '{__name__}'")
    base_lang_name = _get_lang_name(name)
    basecls = get_lang_class(base_lang_name)
    lang_name = base_lang_name + "_torch"
    cls = type(name, (TorchLanguageMixin, basecls), {"lang": lang_name})
    set_lang_class(lang_name, cls)
    return cls
