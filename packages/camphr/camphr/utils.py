"""The utils module defines util functions for internal use."""
from camphr.doc import DocProto, T_Ent, T_Token
from typing import (
    Callable,
    Optional,
    TypeVar,
)

from typing_extensions import Protocol


T_Co = TypeVar("T_Co", covariant=True)


class _SequenceLike(Protocol[T_Co]):
    """Only for type annotation in `binary_search`. Python should provide intersection typing..."""

    def __getitem__(self, idx: int) -> T_Co:
        ...

    def __len__(self) -> int:
        ...


def binary_search(arr: _SequenceLike[T_Co], predicate: Callable[[T_Co], bool]) -> int:
    """Returns minimum index of arr item which satisfies  `predicate`."""
    if not arr or predicate(arr[0]):
        return 0
    left: int = 0
    right = len(arr)
    while right - left > 1:
        m = (right + left) // 2
        if predicate(arr[m]):
            right = m
        else:
            left = m
    return right


def token_from_char_pos(doc: DocProto[T_Token, T_Ent], i: int) -> Optional[T_Token]:
    idx = binary_search(doc, lambda token: token.end_char > i)
    if idx < len(doc):
        token = doc[idx]
        if token.start_char <= i < token.end_char:
            return token
        else:
            return None
    return None
