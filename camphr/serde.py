"""Serialize/Deserialize components"""

from pathlib import Path
from typing import Type, TypeVar
from typing_extensions import Protocol

T = TypeVar("T")


class SerDe(Protocol):
    @classmethod
    def from_disk(cls: Type[T], path: Path) -> T:
        ...

    def to_disk(self, path: Path):
        ...
