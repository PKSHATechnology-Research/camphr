"""The utils module defines util functions used accross sub packages."""
from camphr.doc import DocProto, T_Span
import distutils.spawn
from typing import (
    Callable,
    Optional,
    Protocol,
    TypeVar,
)

from typing_extensions import Literal

from camphr.VERSION import __version__


T_Co = TypeVar("T_Co", covariant=True)


class _SequenceLike(Protocol[T_Co]):
    """Only for type annotation in `binary_search`"""

    def __getitem__(self, idx: int) -> T_Co:
        ...

    def __len__(self) -> int:
        ...


def binary_search(arr: _SequenceLike[T_Co], predicate: Callable[[T_Co], bool]) -> int:
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


def token_from_char_pos(doc: DocProto[T_Span], i: int) -> T_Span:
    idx = binary_search(doc, lambda token: token.l <= i)
    return doc[idx]


def get_juman_command() -> Optional[Literal["juman", "jumanpp"]]:
    for cmd in ["jumanpp", "juman"]:
        if distutils.spawn.find_executable(cmd):
            return cmd  # type: ignore
    return None


T = TypeVar("T")
