"""The utils module defines util functions used accross sub packages."""
from collections import OrderedDict
from pathlib import Path
from typing import Iterable, List

import srsly


class SerializationMixin:
    """Serializes the items in `serialization_fields`

    Example:
        >>> class FooComponent(SerializationMixin, Pipe):
        >>>     serialization_fields = ["bar_attribute"]
        >>> comp = FooComponent(Vocab())
        >>> save_dir = Path("baz_directory")
        >>> comp.to_disk(save_dir) # saved the component into directory
        >>> loaded_comp = spacy.from_disk(save_dir) # load from directory
    """

    def from_bytes(self, bytes_data, exclude=tuple(), **kwargs):
        pkls = srsly.pickle_loads(bytes_data)
        for field in self.serialization_fields:
            setattr(self, field, pkls[field])
        return self

    def to_bytes(self, exclude=tuple(), **kwargs):
        pkls = OrderedDict()
        for field in self.serialization_fields:
            pkls[field] = getattr(self, field, None)
        return srsly.pickle_dumps(pkls)

    def from_disk(self, path: Path, exclude=tuple(), **kwargs):
        path.mkdir(exist_ok=True)
        with (path / f"data.pkl").open("rb") as file_:
            data = file_.read()
        return self.from_bytes(data, **kwargs)

    def to_disk(self, path: Path, exclude=tuple(), **kwargs):
        path.mkdir(exist_ok=True)
        data = self.to_bytes(**kwargs)
        with (path / "data.pkl").open("wb") as file_:
            file_.write(data)


def zero_pad(a: Iterable[List[int]]) -> List[List[int]]:
    """Padding the input so that the lengths of the inside lists are all equal."""
    max_length = max(len(el) for el in a)
    return [el + [0] * (max_length - len(el)) for el in a]
