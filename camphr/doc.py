from typing import Any, Dict, Iterator, List, Optional
from dataclasses import dataclass, field
from typing_extensions import Protocol
from itertools import zip_longest


@dataclass
class Doc:
    tokens: List["Token"]
    is_tagged: bool = False
    user_data: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_words(cls, words: List[str], spaces: List[bool]) -> "Doc":
        tokens: List[Token] = []
        for w, s in zip_longest(words, spaces):
            tokens.append(Token(text=w, whitespace_=s))
        return cls(tokens)

    def __iter__(self) -> Iterator["Token"]:
        for token in self.tokens:
            yield token


@dataclass
class Token:
    text: str
    # trailing whitespce if present
    whitespace_: str
    tag_: Optional[str] = None
    lemma_: Optional[str] = None
    user_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Span:
    user_data: Dict[str, Any] = field(default_factory=dict)
