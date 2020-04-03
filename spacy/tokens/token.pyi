from typing import Any
from spacy.tokens.doc import Doc
from spacy.tokens.underscore import Underscore

class Token:
    text: str
    dep_: str
    tag_: str
    tag: int
    lemma_: str
    pos_: str
    doc: Doc
    _: Underscore
    vector: Any
    i: int
    j: int
    head: Token
    idx: int
    is_sent_start: bool
    @classmethod
    def set_extension(cls, name: str, **kwargs) -> None: ...
    def __len__(self) -> int: ...
