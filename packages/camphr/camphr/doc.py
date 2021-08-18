"""Module doc defines `DocProto` interfaces and its surroundings, and the sample implementations.

Interfaces:
    * DocProto: Unit of pipeline data, simiar to spaCy's `Doc`
    * SpanProto: Span of text, simiar to spaCy's `Doc`
    * TokenProto: Token of text, simiar to spaCy's `Token`, special type of a `SpanProto`
    * EntProto: Ent of text, special type of a `SpanProto`

Implementors:
    * Doc: DocProto
    * Span: SpanProto
    * Token: TokenProto
    * Ent: EntProto

Note:
    It is better for external users to implement their own class that implements the interfaces, not use the implementors as possible.
"""

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
    """Unit of the pipeline data in Camphr. Holding the input text and other parsed results.

    It is better for libraries to depend only on this interface, not on `Doc` directly.
    """

    text: str
    tokens: Optional[List[T_Co]]
    ents: Optional[List[T_Ent]]  # Named Entities
    user_data: Dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, idx: int) -> T_Co:
        ...

    def __len__(self) -> int:
        ...

    def __iter__(self) -> Iterator[T_Co]:
        ...


@dataclass
class Doc(DocProto["Token", "Ent"]):
    """DocProto implementation for internal use."""

    text: str
    tokens: Optional[List["Token"]] = None
    ents: Optional[List["Ent"]] = None
    user_data: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_words(cls, words: List[str]) -> "Doc":
        """Create doc from list of words. The word boundaries (e.g. spaces) must remain kept."""
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
        """Returns the number of tokens in doc. Raise if self is not tokenized."""
        return len(unwrap(self.tokens))

    def __iter__(self) -> Iterator["Token"]:
        tokens = unwrap(self.tokens)
        for token in tokens:
            yield token


class SpanProto(UserDataProto, Protocol):
    """Span of Doc, similar to spaCy's span.

    Attributes:
        start_char: start boundary in doc text (inclusive)
        end_char: end boundary in doc text (exclusive)
    """

    start_char: int  # left boundary in doc
    end_char: int  # right boundary in doc
    doc: Doc

    @property
    def text(self) -> str:
        ...


@dataclass
class Span(SpanProto):
    """A SpanProto implementation for internal use."""

    start_char: int  # left boundary in doc
    end_char: int  # right boundary in doc
    doc: Doc
    user_data: Dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return self.doc.text[self.start_char : self.end_char]


class TokenProto(SpanProto):
    """`TokenProto` is a special type of `SpanProto`. It may contain `tag_` and `lemma_` information.

    Attributes:
        tag_: May contain `tag` information of this token. The underline after the field name is for compatibility with spaCy.
        lemma_: May contain `lemma` information of this token. The underline after the field name is for compatibility with spaCy.
    """

    tag_: Optional[str]
    lemma_: Optional[str]


@dataclass
class Token(TokenProto, Span):
    """A TokenProto implementation for internal use."""

    doc: Doc
    tag_: Optional[str] = None
    lemma_: Optional[str] = None


class EntProto(SpanProto):
    """`EntProto` is a representation of Named Entity Span, and a special type of `SpanProto`.

    Attributes:
        label: Label of this ent, e.g. `PERSON`
        score: May contain float score value
    """

    label: Optional[str]
    score: Optional[float]


@dataclass
class Ent(EntProto, Span):
    """A EntProto implementation for internal use."""

    label: Optional[str] = None
    score: Optional[float] = None
