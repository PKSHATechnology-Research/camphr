from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
    overload
)

from spacy import tokens
from spacy.tokens.underscore import Underscore
from spacy.vocab import Vocab

from ._retokenize import Retokenizer
from .span import Span
from .token import Token

class Doc:
    text: str
    doc: "Doc"
    is_parsed: bool
    is_tagged: bool
    is_sentenced: bool
    sents: Iterator[tokens.Span]
    user_data: Dict[Union[str, Tuple], Any]
    tensor: Any
    user_hooks: Dict[str, Callable]
    user_span_hooks: Dict[str, Callable]
    user_token_hooks: Dict[str, Callable]
    _: Underscore
    cats: Dict[str, Any]
    vector: Any
    vocab: Vocab
    noun_chunks_iterator: Callable[[Doc], Iterable[Tuple[int, int, str]]]
    noun_chunks: Iterator[Span]
    is_nered: bool
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
        label: Union[int, str]= 0,
        kb_id: int = 0,
        vector: Any = None,
    ): ...
    def retokenize(self) -> Retokenizer: ...
    @property
    def ents(self) -> Tuple[Span, ...]: ...
    @ents.setter
    def ents(self, ents: Iterable[Span]): ...
    def to_array(self, attrs: List):...
    def from_array(self, attrs: List, array: List):...
