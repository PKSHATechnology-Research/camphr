from typing_extensions import Protocol
from typing import TypeVar


class Doc(Protocol):
    ...


T_Doc = TypeVar("T_Doc", bound=Doc)


class Token(Protocol):
    ...


class Span(Protocol):
    ...
