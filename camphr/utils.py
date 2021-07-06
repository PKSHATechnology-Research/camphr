"""The utils module defines util functions used accross sub packages."""
import bisect
from camphr.doc import Doc, DocProto, Span, TokenProto
import distutils.spawn
import importlib
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    Union,
    cast,
)

from typing_extensions import Literal

from camphr.VERSION import __version__


T = TypeVar("T")


class _SequenceLike(Protocol[T]):
    """Only for type annotation in `binary_search`"""

    def __getitem__(self, idx: int) -> T:
        ...

    def __len__(self) -> int:
        ...


def binary_search(arr: _SequenceLike[T], predicate: Callable[[T], bool]) -> int:
    """Returns minimum index of arr item which satisfies  `predicate`"""
    if not arr or predicate(arr[0]):
        return 0
    ng: int = 0
    ok = len(arr)
    while ok - ng > 1:
        m = (ok + ng) // 2
        if predicate(arr[m]):
            ok = m
        else:
            ng = m
    return ok


def token_from_char_pos(doc: DocProto, i: int) -> TokenProto:
    idx = binary_search(doc, lambda token: token.l <= i)
    return doc[idx]


def get_juman_command() -> Optional[Literal["juman", "jumanpp"]]:
    for cmd in ["jumanpp", "juman"]:
        if distutils.spawn.find_executable(cmd):
            return cmd  # type: ignore
    return None
