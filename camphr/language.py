from typing import Callable, List
from camphr.doc import Doc
from typing_extensions import Protocol


class PipeProto(Protocol):
    def __call__(self, doc: Doc) -> Doc:
        ...

    def to_disk(self, path: str):
        ...

    def from_disk(self, path: str):
        ...


class LanguageProto(Protocol):
    pipeline: List[PipeProto]
    tokenizer: Callable[[str], Doc]

    def __call__(self, text: str) -> Doc:
        ...

    def to_disk(self, path: str):
        ...

    def from_disk(self, path: str):
        ...

    def make_doc(self, text: str) -> Doc:
        return self.tokenizer(text)
