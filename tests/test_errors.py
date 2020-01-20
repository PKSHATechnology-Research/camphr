import pytest

from camphr.errors import Warnings


def test_warning():
    with pytest.warns(RuntimeWarning, match="Foo 1 wow 2!"):
        Warnings._W_FOR_TEST(1, bar=2)
