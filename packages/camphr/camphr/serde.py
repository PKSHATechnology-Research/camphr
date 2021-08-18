"""Serialize/Deserialize implementations.

Pipeline impementors should also imlement `SerDe` interace so that it can be serialized and deserialized into/from disk.

Examples:
    >>> class FooPipe:
            def from_disk(self, ...): ...
            def to_disk(self, ...): ...
    >>> pipe = FooPipe()
    >>> nlp = YourPipeline(pipes = [pipe])
    >>> path =  "/path/to/save"
    >>> camphr.serde.to_disk(nlp, path) # save into disk
    >>> nlp = camphr.serde.from_disk(path) # load from disk
"""

from dataclasses import dataclass, asdict
import importlib
import dataclass_utils.error
import pickle

from pathlib import Path
from typing import Any, ClassVar, List, Tuple, Type, TypeVar
import dataclass_utils
from typing_extensions import Protocol, runtime_checkable
import json


T = TypeVar("T")


@runtime_checkable
class SerDe(Protocol):
    """Interface used for `camphr.from_disk` and `camphr.to_disk` functions."""

    @classmethod
    def from_disk(cls: Type[T], path: Path) -> T:
        raise ValueError("Not implemented")

    def to_disk(self, path: Path) -> None:
        raise ValueError("Not implemented")


def to_disk(obj: "SerDe", path: Path):
    """Save `SerDe` object into `path` directory"""
    if not isinstance(obj, SerDe) or isinstance(obj, type):
        raise ValueError(f"{obj} doesn't implement `SerDe`")

    path.mkdir(exist_ok=True)
    # write metadata so that `from_disk` can get the class of `obj`
    module_name, class_name = _get_fullname(obj.__class__)
    meta = _Meta(module_name, class_name)
    meta.to_disk(path)
    # delegate to obj
    obj.to_disk(path)


def from_disk(path: Path) -> "SerDe":
    """Load obj from `path` directory"""
    # load metadata
    meta = _Meta.from_disk(path)
    kls = _get_class(meta.module_name, meta.class_name)
    if not isinstance(kls, SerDe):
        raise ValueError(f"Invalid class loaded: `{kls}` doesn't implement SerDe")
    # delegate to kls
    return kls.from_disk(path)


def _get_fullname(kls: Type[Any]) -> Tuple[str, str]:
    """Get fully qualified name of a class for ser/de"""
    mod = kls.__module__
    class_name = kls.__name__
    return mod, class_name


def _get_class(module_name: str, class_name: str) -> Type[Any]:
    """Get class object by string name"""
    return getattr(importlib.import_module(module_name), class_name)


T_SerdeDataclass = TypeVar("T_SerdeDataclass", bound="SerDeDataclassMixin")


class SerDeDataclassMixin(SerDe):
    """SerDe Mixin for dataclass. dataclasses which can be represented as JSON can implement `SerDe` simply by inheriting this Mixin.

    >>> @dataclass
        class DataClassFoo(SerDeDataclassMixin):
            ...
    >>> d = DataClassFoo()
    >>> camphr.serde.to_disk(d, "path/to/save")
    """

    FILENAME: ClassVar[str]  # JSON file name.

    def to_disk(self, path: Path):
        """Save self as json."""
        meta_path = path / self.FILENAME
        meta_path.write_text(json.dumps(asdict(self)))

    @classmethod
    def from_disk(cls: Type[T_SerdeDataclass], path: Path) -> T_SerdeDataclass:
        """Load cls from json file."""
        meta_path = path / cls.FILENAME
        try:
            data = dataclass_utils.into(json.loads(meta_path.read_text()), cls)
        except dataclass_utils.error.Error as e:
            raise ValueError("Invalid metadata content.") from e
        return data


@dataclass
class _Meta(SerDeDataclassMixin):
    """Metadata for `to_disk` and `from_disk`."""

    module_name: str
    class_name: str
    FILENAME: ClassVar[str] = "camphr_serialization_meta.json"


T_Ser = TypeVar("T_Ser", bound="SerializationMixin")


class SerializationMixin(SerDe):
    """SerDe Mixin for arbitrary class. Any class can implement `SerDe` if their fields to be saved can be dumped as pickle.

    Attributes:
        serialization_fields: ClassVar for indicating which fields to be saved.
    Examples:
        >>> class Foo(SerializationMixin):
                serialization_fields = ["foo"]
                def __init__(self):
                    self.foo = 'foo' # this field will be saved.
                    self.bar = 'bar' # but this is not because not listed in `serialization_fields`.
        >>> obj = Foo()
        >>> camphr.serde.to_disk(obj, "path/to/save")
    """

    serialization_fields: List[str] = []

    @classmethod
    def from_disk(cls: Type[T_Ser], path: Path) -> T_Ser:
        """Save fields in `serialization_fields` as pickle."""
        data = {}
        for k in cls.serialization_fields:
            data[k] = pickle.loads((path / k).read_bytes())
        return cls(**data)  # type: ignore

    def to_disk(self, path: Path):
        """Load fields from pickle file, then create the instance."""
        path.mkdir(exist_ok=True)
        for k in self.serialization_fields:
            data = getattr(self, k)
            with (path / k).open("wb") as f:
                pickle.dump(data, f)
