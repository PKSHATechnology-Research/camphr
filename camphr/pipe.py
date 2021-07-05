from camphr.doc import DocProto
from typing import Protocol


class Pipe(Protocol):
    def __call__(self, doc: DocProto) -> DocProto:
        ...
