from typing import (
    Any,
    Callable,
    Iterable,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    overload,
    Sequence,
)
from spacy.tokens.underscore import Underscore

from spacy.vocab import Vocab
from .span import Span

from .token import Token
from ._retokenize import Retokenizer

from spacy import tokens

class Doc:
    text: str
    is_parsed: bool
    is_tagged: bool
    sents: Iterator[tokens.Span]
    user_data: Dict[str, Any]
    tensor: Any
    user_hooks: Dict[str, Callable]
    user_span_hooks: Dict[str, Callable]
    user_token_hooks: Dict[str, Callable]
    _: Underscore
    cats: Dict[str, Any]
    vector: Any
    def __init__(
        self,
        vocab: Vocab,
        words: Optional[List[str]] = None,
        spaces: Optional[List[bool]] = None,
        user_data: Any = None,
        orths_and_spaces: Any = None,
    ): ...
    def __iter__(self) -> Iterator[Token]: ...
    def __len__(self) -> int: ...
    @overload
    def __getitem__(self, i: int) -> Token: ...
    @overload
    def __getitem__(self, i: slice) -> Span: ...
    @classmethod
    def set_extension(cls, name: str, **kwargs) -> None: ...
    def char_span(
        self,
        start_idx: int,
        end_idx: int,
        label: int = 0,
        kb_id: int = 0,
        vector: Any = None,
    ): ...
    def retokenize(self) -> Retokenizer: ...
    @property
    def ents(self) -> Tuple[Span, ...]: ...
    @ents.setter
    def ents(self, ents: Iterable[Span]): ...
