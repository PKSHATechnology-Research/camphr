from camphr.utils import unwrap
from typing import Any, Dict, Iterator, List, Optional, Protocol, TypeVar
from dataclasses import dataclass, field


T_Token = TypeVar("T_Token", bound="TokenProto", covariant=True)
T_Span = TypeVar("T_Span", bound="SpanProto", covariant=True)


class UserDataProto(Protocol):
    user_data: Dict[str, Any]


class DocProto(UserDataProto, Protocol):
    """Doc interface"""

    text: str
    tokens: Optional[List["TokenProto"]]
    ents: Optional[List["EntProto"]]

    def __getitem__(self, idx: int) -> "TokenProto":
        return unwrap(self.tokens)[idx]

    def __len__(self) -> int:
        return len(unwrap(self.tokens))

    def __iter__(self) -> Iterator["TokenProto"]:
        tokens = unwrap(self.tokens)
        for token in tokens:
            yield token


@dataclass
class Doc(DocProto):
    text: str
    tokens: Optional[List["TokenProto"]] = None
    ents: Optional[List["EntProto"]] = None
    user_data: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_words(cls, words: List[str]) -> "Doc":
        doc = cls("".join(words))
        tokens: List[TokenProto] = []
        l = 0
        for w in words:
            r = l + len(w)
            tokens.append(Token(l, r, doc))
            l = r
        doc.tokens = tokens
        return doc


class SpanProto(UserDataProto, Protocol):
    """Span interface"""

    l: int  # left boundary in doc
    r: int  # right boundary in doc
    doc: DocProto

    @property
    def text(self) -> str:
        return self.doc.text[self.l : self.r]


@dataclass
class Span(SpanProto, UserDataProto):
    l: int  # left boundary in doc
    r: int  # right boundary in doc
    doc: DocProto
    user_data: Dict[str, Any] = field(default_factory=dict)


class TokenProto(SpanProto):
    """Token interface"""

    tag_: Optional[str]
    lemma_: Optional[str]


@dataclass
class Token(Span, TokenProto):
    tag_: Optional[str] = None
    lemma_: Optional[str] = None


class EntProto(SpanProto):
    label: str
    score: Optional[float]


@dataclass
class Ent(EntProto):
    l: int
    r: int
    doc: DocProto
    label: str
    score: float
    user_data: Dict[str, Any] = field(default_factory=dict)
