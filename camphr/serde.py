"""Serialize/Deserialize components"""
from dataclasses import dataclass, asdict
import importlib
import dataclass_utils.error
import pickle

from pathlib import Path
from typing import Any, ClassVar, List, Tuple, Type, TypeVar, runtime_checkable
import dataclass_utils
from typing_extensions import Protocol
import json


T = TypeVar("T")

# use runtime_checkable here for `from_disk`
@runtime_checkable
class SerDe(Protocol):
    @classmethod
    def from_disk(cls: Type[T], path: Path) -> T:
        raise ValueError("Not implemented")

    def to_disk(self, path: Path) -> None:
        raise ValueError("Not implemented")


def to_disk(obj: "SerDe", path: Path):
    """Dump obj into `path` directory"""
    if not isinstance(obj, SerDe) or isinstance(obj, type):  # type: ignore
        raise ValueError(f"{obj} doesn't implement `SerDe`")

    path.mkdir(exist_ok=True)
    # write metadata so that `from_disk` can get class name of `obj`
    module_name, class_name = _get_fullname(obj.__class__)
    meta = Meta(module_name, class_name)
    meta.to_disk(path)
    # delegate to obj
    obj.to_disk(path)


def from_disk(path: Path) -> "SerDe":
    """Load obj from `path` directory"""
    # load metadata
    meta = Meta.from_disk(path)
    kls = _get_class(meta.module_name, meta.class_name)
    if not isinstance(kls, SerDe):
        raise ValueError(f"Invalid class loaded: `{kls}` doesn't implement SerDe")
    # delegate to kls
    return kls.from_disk(path)


def _get_fullname(kls: Type[Any]) -> Tuple[str, str]:
    """Get fully qualified name of a class for ser/deserialization"""
    mod = kls.__module__
    class_name = kls.__name__
    return mod, class_name


def _get_class(module_name: str, class_name: str) -> Type[Any]:
    return getattr(importlib.import_module(module_name), class_name)


T_SerdeDataclass = TypeVar("T_SerdeDataclass", bound="SerDeDataclassMixin")


class SerDeDataclassMixin(SerDe):
    FILENAME: ClassVar[str]

    def to_disk(self, path: Path):
        meta_path = path / self.FILENAME
        meta_path.write_text(json.dumps(asdict(self)))

    @classmethod
    def from_disk(cls: Type[T_SerdeDataclass], path: Path) -> T_SerdeDataclass:
        meta_path = path / cls.FILENAME
        try:
            data = dataclass_utils.into(json.loads(meta_path.read_text()), cls)
        except dataclass_utils.error.Error as e:
            raise ValueError(f"Invalid metadata content. ") from e
        return data


@dataclass
class Meta(SerDeDataclassMixin):
    """Metadata for `to_disk` and `from_disk`"""

    module_name: str
    class_name: str
    FILENAME: ClassVar[str] = "camphr_serialization_meta.json"


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
