from camphr.doc import T_Doc
from typing_extensions import Protocol


class Pipe(Protocol):
    """Interface like spaCy.Pipe, simply taking a `Doc` and returns a `Doc`."""

    def __call__(self, doc: T_Doc) -> T_Doc:
        ...
