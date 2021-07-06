from camphr.serde import from_disk, get_fullname, SerDe, Path, to_disk
import pytest
from typing import Any, Tuple, Type
from .utils import Dummy, DummySerde


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


@pytest.mark.parametrize("obj", [DummySerde()])
def test_serde_toplevel(tmpdir: str, obj: Any):
    path = Path(tmpdir)
    to_disk(obj, path)
    deserialized = from_disk(path)

    assert type(obj) is type(deserialized)


@pytest.mark.parametrize("obj", [Dummy])
def test_serde_toplevel_error(tmpdir: str, obj: Any):
    path = Path(tmpdir)
    with pytest.raises(ValueError):
        to_disk(obj, path)
