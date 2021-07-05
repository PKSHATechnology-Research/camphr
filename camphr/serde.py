"""Serialize/Deserialize components"""
import pickle

from pathlib import Path
from typing import List, Type, TypeVar
from typing_extensions import Protocol

T = TypeVar("T")


class SerDe(Protocol):
    @classmethod
    def from_disk(cls: Type[T], path: Path) -> T:
        ...

    def to_disk(self, path: Path):
        ...


T_Ser = TypeVar("T_Ser", bound="SerializationMixin")


class SerializationMixin(SerDe):
    serialization_fields: List[str] = []

    @classmethod
    def from_disk(cls: Type[T_Ser], path: Path) -> T_Ser:
        data = {}
        for k in cls.serialization_fields:
            data[k] = pickle.loads((path / k).read_bytes())
        return cls(**data)  # type: ignore

    def to_disk(self, path: Path):
        path.mkdir(exist_ok=True)
        for k in self.serialization_fields:
            data = getattr(self, k)
            with (path / k).open("wb") as f:
                pickle.dump(data, f)
