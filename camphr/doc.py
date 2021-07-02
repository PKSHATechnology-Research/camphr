from typing import Any, Dict
from dataclasses import dataclass, field
from typing_extensions import Protocol


class DocProto(Protocol):
    user_data: Dict[str, Any]


class Doc(DocProto):
    user_data: Dict[str, Any]


class TokenProto(Protocol):
    user_data: Dict[str, Any]


@dataclass
class Token(TokenProto):
    user_data: Dict[str, Any] = field(default_factory=dict)


class SpanProto(Protocol):
    user_data: Dict[str, Any]


class Span(SpanProto):
    user_data: Dict[str, Any] = field(default_factory=dict)
