from camphr.doc import Doc
from typing import Protocol


class Pipe(Protocol):
    def __call__(self, doc: Doc) -> Doc:
        ...
