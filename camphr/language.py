from camphr.doc import DocProto
from typing_extensions import Protocol


class LanguageProto(Protocol):
    def __call__(self, text: str) -> DocProto:
        ...
