from typing import Protocol
from camphr.doc import DocProto, T_Span


class Nlp(Protocol):
    def __call__(self, text: str) -> DocProto[T_Span]:
        ...
