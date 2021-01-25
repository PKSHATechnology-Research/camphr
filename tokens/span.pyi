from typing import Any, Iterator, Tuple, Union, overload

from spacy.tokens.doc import Doc
from spacy.tokens.token import Token
from spacy.tokens.underscore import Underscore

class Span:
    text: str
    doc: Doc
    ents: Tuple[Span]
    vector: Any
    start: int
    end: int
    sent: "Span"
    start_char: int
    end_char: int
    label: int
    label_: str
    def __iter__(self) -> Iterator[Token]: ...
    def __len__(self) -> int: ...
    _: Underscore
    @overload
    def __getitem__(self, i: int) -> Token: ...
    @overload
    def __getitem__(self, i: slice) -> Span: ...
    @classmethod
    def set_extension(cls, name: str, **kwargs) -> None: ...
    def __init__(
        self,
        doc: Doc,
        start: int,
        end: int,
        label: Union[int, str] = 0,
        vector: Any = None,
        vector_norm: Any = None,
        kb_id: int = 0,
    ): ...
