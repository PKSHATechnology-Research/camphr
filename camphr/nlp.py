from typing_extensions import Protocol
from camphr.doc import DocProto, T_Ent, T_Token


class Nlp(Protocol):
    def __call__(self, text: str) -> DocProto[T_Token, T_Ent]:
        ...
