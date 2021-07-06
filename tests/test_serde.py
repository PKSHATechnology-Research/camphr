from camphr.serde import get_fullname, SerDe, Path
import pytest
from typing import Any, Tuple, Type
from .utils import Dummy


@pytest.mark.parametrize(
    "kls,expected",
    [
        (Dummy, ("tests.utils", "Dummy")),
        (SerDe, ("camphr.serde", "SerDe")),
        (Path, ("pathlib", "Path")),
    ],
)
def test_get_fullname(kls: Type[Any], expected: Tuple[str, str]):
    assert get_fullname(kls) == expected
