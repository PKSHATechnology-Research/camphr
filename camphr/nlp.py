from typing import Protocol
from camphr.doc import DocProto


class Nlp(Protocol):
    def __call__(self, text: str) -> DocProto:
        ...
