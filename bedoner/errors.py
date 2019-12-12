import warnings


class W:
    def __init__(self, msg: str, warning_cls: Warning):
        self.msg = msg
        self.warning_cls = warning_cls

    def __call__(self, *args, **kwargs):
        warnings.warn(self.msg.format(*args, **kwargs), self.warning_cls)


class Warnings:
    _W_FOR_TEST = W("Foo {} wow {bar}!", RuntimeWarning)
    W0 = W("{} has been deprecated. Please use {} instead.", DeprecationWarning)
    W1 = W("{} has been deprecated.", DeprecationWarning)
