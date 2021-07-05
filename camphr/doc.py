from typing import Any, Dict, Iterator, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class Doc:
    text: str
    is_tagged: bool = False
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

    def __iter__(self) -> Iterator["Token"]:
        for token in self.tokens:
            yield token


@dataclass
class Span:
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
