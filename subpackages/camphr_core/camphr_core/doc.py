from typing import Any, Iterable, Iterator, TypeVar, Union, overload
from typing_extensions import Protocol


T_Token = TypeVar("T_Token", bound="Token", covariant=True)
T_Span = TypeVar("T_Span", covariant=True)


class Token(Protocol):
    """Interface for spacy.Token"""

    idx: int


class Span(Protocol):
    """Interface for spacy.Span"""

    ...


class Doc(Protocol[T_Token, T_Span]):
    """Interface for spacy.Doc"""

    sents: Iterable[T_Span]

    def __iter__(self) -> Iterator[T_Token]:
        ...

    @overload
    def __getitem__(self, i: int) -> T_Token:
        ...

    @overload
    def __getitem__(self, i: slice) -> T_Span:
        ...

    def __getitem__(self, i) -> T_Span:
        raise NotImplementedError()

    def char_span(
        self,
        start_idx: int,
        end_idx: int,
        label: Union[int, str] = 0,
        kb_id: int = 0,
        vector: Any = None,
    ) -> Span:
        ...
