from typing import Any, Iterator

from spacy.tokens.doc import Doc
from spacy.tokens.underscore import Underscore

class Token:
    text: str
    dep_: str
    tag_: str
    tag: int
    lemma_: str
    whitespace_: str
    is_space: bool
    pos_: str
    pos: int
    doc: Doc
    _: Underscore
    vector: Any
    i: int
    j: int
    head: Token
    idx: int
    is_sent_start: bool
    rights: Iterator["Token"]
    lefts: Iterator["Token"]
    ancestors: Iterator["Token"]
    children: Iterator["Token"]
    subtree: Iterator["Token"]
    @classmethod
    def set_extension(cls, name: str, **kwargs: Any) -> None: ...
    def __len__(self) -> int: ...
