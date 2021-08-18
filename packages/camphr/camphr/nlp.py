from typing_extensions import Protocol
from camphr.doc import DocProto, T_Ent, T_Token


class Nlp(Protocol):
    """Interface similar to spaCy.Language, simply representing a class that takes a text and output a `Doc`"""

    def __call__(self, text: str) -> DocProto[T_Token, T_Ent]:
        ...
