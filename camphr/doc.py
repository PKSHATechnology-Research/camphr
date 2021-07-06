from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Optional,
    Protocol,
    TypeVar,
    runtime_checkable,
)


T = TypeVar("T")
T_Co = TypeVar("T_Co", covariant=True)
T_Span = TypeVar("T_Span", bound="SpanProto")
T_Doc = TypeVar("T_Doc", bound="DocProto")  # type: ignore


class UserDataProto(Protocol):
    user_data: Dict[str, Any]


def unwrap(v: Optional[T]) -> T:
    if v is None:
        raise ValueError(f"{v} is None")
    return v


@runtime_checkable
class DocProto(Protocol[T_Co]):
    text: str

    def __getitem__(self, idx: int) -> T_Co:
        ...

    def __len__(self) -> int:
        ...

    def __iter__(self) -> Iterator[T_Co]:
        ...


@dataclass
class Doc(DocProto["Token"]):
    text: str
    tokens: Optional[List["Token"]] = None
    ents: Optional[List["Ent"]] = None
    user_data: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_words(cls, words: List[str]) -> "Doc":
        doc = cls("".join(words))
        tokens: List[Token] = []
        l = 0
        for w in words:
            r = l + len(w)
            tokens.append(Token(l, r, doc))
            l = r
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


class SpanProto(Protocol):
    l: int  # left boundary in doc
    r: int  # right boundary in doc
    doc: Doc


@dataclass
class Span(UserDataProto, SpanProto):
    l: int  # left boundary in doc
    r: int  # right boundary in doc
    doc: Doc
    user_data: Dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return self.doc.text[self.l : self.r]


@dataclass
class Token(Span):
    tag_: Optional[str] = None
    lemma_: Optional[str] = None


@dataclass
class Ent(UserDataProto, SpanProto):
    l: int
    r: int
    doc: Doc
    label: str
    score: float
    user_data: Dict[str, Any] = field(default_factory=dict)
