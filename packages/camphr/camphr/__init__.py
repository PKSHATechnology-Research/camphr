"""Package `camphr` provides several interfaces and utilities for other subpackages:

- nlp.py: `Nlp` interface. Nlp may contain multiple `Pipe`s
- pipe.py: `Pipe` interface. Simply `Doc -> Doc`
- doc.py: `Doc` interface and implementations
- serde.py: Serialization/Deserialization supports

Typical users would implement `Nlp` and `SerDe`: a class takes a text and outputs a `DocProto`, and can be serialized/deserialized via `to_disk` and `from_disk`.
All of the above entities are not concrete classes but interfaces that allow users to implement their own classes without inheriting anything.
"""
from .VERSION import __version__

__all__ = ["__version__"]
