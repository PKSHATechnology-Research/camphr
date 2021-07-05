from typing import Any, Dict, Iterator, List, Optional, Protocol, TypeVar
from dataclasses import dataclass, field


T_Token = TypeVar("T_Token", bound="TokenProto", covariant=True)


class UserDataProto(Protocol):
    user_data: Dict[str, Any]


class DocProto(UserDataProto, Protocol[T_Token]):
    """Doc interface"""

    text: str
    tokens: Optional[List[T_Token]]

    def __iter__(self) -> Iterator[T_Token]:
        if self.tokens is None:
            raise ValueError("doc.tokens is None")
        for token in self.tokens:
            yield token


@dataclass
class Doc(DocProto["Token"]):
    text: str
    tokens: Optional[List["Token"]] = None
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


class SpanProto(UserDataProto, Protocol):
    """Span interface"""

    l: int  # left boundary in doc
    r: int  # right boundary in doc
    doc: Doc

    @property
    def text(self) -> str:
        return self.doc.text[self.l : self.r]


@dataclass
class Span(SpanProto):
    l: int  # left boundary in doc
    r: int  # right boundary in doc
    doc: Doc
    user_data: Dict[str, Any] = field(default_factory=dict)


class TokenProto(SpanProto):
    """Token interface"""

    tag_: Optional[str]
    lemma_: Optional[str]


@dataclass
class Token(Span, TokenProto):
    tag_: Optional[str] = None
    lemma_: Optional[str] = None
