from bedoner.utils import zero_pad, inject_mixin
import bedoner.lang.juman as juman
import bedoner.lang.mecab as mecab
from bedoner.lang.torch_mixin import TorchLanguageMixin


def test_zero_pad():
    a = [[1, 2], [2, 3, 4]]
    b = [[1, 2, 0], [2, 3, 4]]
    assert b == zero_pad(a)


def test_inject_mixin():
    cls = inject_mixin(TorchLanguageMixin, juman.Japanese)
    assert issubclass(cls, juman.Japanese)

    cls = inject_mixin(TorchLanguageMixin, mecab.Japanese)
    assert issubclass(cls, mecab.Japanese)
