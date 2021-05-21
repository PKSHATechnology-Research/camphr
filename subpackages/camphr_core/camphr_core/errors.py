import warnings
from typing import Any, Type


class W:
    def __init__(self, msg: str, warning_cls: Type[Warning]):
        self.msg = msg
        self.warning_cls = warning_cls

    def __call__(self, *args: Any, **kwargs: Any):
        warnings.warn(self.msg.format(*args, **kwargs), self.warning_cls, stacklevel=3)


class E:
    def __init__(self, msg: str, exception_cls: Type[Exception]):
        self.msg = msg
        self.exception_cls = exception_cls

    def __call__(self, *args: Any, **kwargs: Any):
        raise self.exception_cls(self.msg.format(*args, **kwargs))


class Warnings:
    _W_FOR_TEST = W("Foo {} wow {bar}!", RuntimeWarning)
    W0 = W("{} has been deprecated. Please use {} instead.", DeprecationWarning)
    W1 = W("{} has been deprecated.", DeprecationWarning)


class Errors:
    E0 = E(
        "Requires {package}. Please install it with `pip install {package}`.",
        ImportError,
    )
