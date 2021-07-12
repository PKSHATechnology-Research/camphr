from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Optional,
    TypeVar,
)
from typing_extensions import Protocol, runtime_checkable


T = TypeVar("T")
T_Co = TypeVar("T_Co")
T_Span = TypeVar("T_Span", bound="SpanProto")
T_Ent = TypeVar("T_Ent", bound="EntProto")
T_Token = TypeVar("T_Token", bound="TokenProto")
T_Doc = TypeVar("T_Doc", bound="DocProto")  # type: ignore


class UserDataProto(Protocol):
    user_data: Dict[str, Any]


def unwrap(v: Optional[T]) -> T:
    if v is None:
        raise ValueError(f"{v} is None")
    return v


@runtime_checkable
class DocProto(UserDataProto, Protocol[T_Co, T_Ent]):
    text: str
    tokens: Optional[List[T_Co]]
    ents: Optional[List[T_Ent]]
    user_data: Dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, idx: int) -> T_Co:
        ...

    def __len__(self) -> int:
        ...

    def __iter__(self) -> Iterator[T_Co]:
        ...


@dataclass
class Doc(DocProto["Token", "Ent"]):
    text: str
    tokens: Optional[List["Token"]] = None
    ents: Optional[List["Ent"]] = None
    user_data: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_words(cls, words: List[str]) -> "Doc":
        doc = cls("".join(words))
        tokens: List[Token] = []
        left = 0
        for w in words:
            right = left + len(w)
            tokens.append(Token(left, right, doc))
            left = right
        doc.tokens = tokens
        return doc

    def __getitem__(self, idx: int) -> "Token":
        return unwrap(self.tokens)[idx]

    def __len__(self) -> int:
        return len(unwrap(self.tokens))

    def __iter__(self) -> Iterator["Token"]:
        tokens = unwrap(self.tokens)
        for token in tokens:
            yield token


class SpanProto(UserDataProto, Protocol):
    start_char: int  # left boundary in doc
    end_char: int  # right boundary in doc
    doc: Doc

    @property
    def text(self) -> str:
        ...


@dataclass
class Span(SpanProto):
    start_char: int  # left boundary in doc
    end_char: int  # right boundary in doc
    doc: Doc
    user_data: Dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return self.doc.text[self.start_char : self.end_char]


class TokenProto(SpanProto):
    tag_: Optional[str]
    lemma_: Optional[str]


@dataclass
class Token(TokenProto, Span):
    doc: Doc
    tag_: Optional[str] = None
    lemma_: Optional[str] = None


class EntProto(SpanProto):
    label: Optional[str]
    score: Optional[float]


@dataclass
class Ent(EntProto, Span):
    label: Optional[str] = None
    score: Optional[float] = None
