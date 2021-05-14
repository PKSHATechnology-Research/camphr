"""The utils module defines util functions used accross sub packages."""
import bisect
from collections import OrderedDict
import distutils.spawn
import importlib
import json
from pathlib import Path
import re
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)
from typing_extensions import Literal

from more_itertools import padded
import spacy
from spacy.errors import Errors
from spacy.language import BaseDefaults
from spacy.tokens import Doc, Span, Token
from spacy.util import filter_spans
from toolz import curry
import yaml

from camphr.VERSION import __version__
from camphr.types import Pathlike


def zero_pad(a: List[List[int]], pad_value: int = 0) -> List[List[int]]:
    """Padding the input so that the lengths of the inside lists are all equal."""
    if len(a) == 0:
        return []
    max_length = max(len(el) for el in a)
    if max_length == 0:
        return a
    return [list(padded(el, fillvalue=pad_value, n=max_length)) for el in a]


RE_URL = re.compile(
    r"https?://(www\.)?[-a-zA-Z0-9@:%._+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&/=]*)"
)


def token_from_char_pos(doc: Doc, i: int) -> Token:
    token_idxs = [t.idx for t in doc]
    return doc[bisect.bisect(token_idxs, i) - 1]


def _get_covering_span(doc: Doc, i: int, j: int, **kwargs: Any) -> Span:
    token_idxs = [t.idx for t in doc]
    i = bisect.bisect(token_idxs, i) - 1
    j = bisect.bisect_left(token_idxs, j)
    return Span(doc, i, j, **kwargs)


def destruct_token(doc: Doc, *char_pos: int) -> Doc:
    for i in char_pos:
        with doc.retokenize() as retokenizer:
            token = token_from_char_pos(doc, i)
            heads = [token] * len(token)
            retokenizer.split(doc[token.i], list(token.text), heads=heads)
    return doc


def get_doc_char_span(
    doc: Doc,
    i: int,
    j: int,
    destructive: bool = True,
    covering: bool = False,
    **kwargs: Any,
) -> Optional[Span]:
    """Get Span from Doc with char position, similar to doc.char_span.

    Args:
        i: The index of the first character of the span
        j: The index of the first character after the span
        destructive: If True, tokens in [i,j) will be splitted and make sure to return span.
        covering: If True, [i,j) will be adjusted to match the existing token boundaries. It precedes `destructive`.
        kwargs: passed to Doc.char_span
    """
    span = doc.char_span(i, j, **kwargs)
    if not span and covering:
        span = _get_covering_span(doc, i, j, **kwargs)
    if not span and destructive:
        destruct_token(doc, i, j)
        span = doc.char_span(i, j, **kwargs)
    return span


def get_doc_char_spans_list(
    doc: Doc, spans: Iterable[Tuple[int, int]], destructive: bool = True, **kwargs: Any
) -> List[Span]:
    res: List[Span] = []
    for i, j in spans:
        span = get_doc_char_span(doc, i, j, destructive=destructive, **kwargs)
        if span:
            res.append(span)
    return res


def merge_spans(doc: Doc, spans: Iterable[Span]):
    spans = filter_spans(spans)
    with doc.retokenize() as retokenizer:
        for span in spans:
            retokenizer.merge(span)


def split_keepsep(text: str, sep: str):
    texts = text.split(sep)
    if len(texts) == 1:
        return [text]

    res = [t + sep for t in texts[:-1]]
    last = texts[-1]
    if len(last):
        if text.endswith(sep):
            last += sep
        res.append(last)
    return res


def import_attr(import_path: str) -> Any:
    items = import_path.split(".")
    module_name = ".".join(items[:-1])
    return getattr(importlib.import_module(module_name), items[-1])


def get_requirements_line():
    return f"camphr>={__version__}"


def get_defaults(lang: str) -> Type[BaseDefaults]:
    try:
        lang_cls = spacy.util.get_lang_class(lang)  # type: ignore
    except Exception:
        return BaseDefaults
    return getattr(lang_cls, "Defaults", BaseDefaults)


def get_labels(labels_or_path: Union[List[str], Pathlike]) -> List[str]:
    if isinstance(labels_or_path, (str, Path)):
        path = Path(labels_or_path)
        if path.suffix == ".json":
            return json.loads(path.read_text())
        elif path.suffix in {".yml", ".yaml"}:
            return yaml.safe_load(path.read_text())
    return cast(List[str], labels_or_path)


def get_by_dotkey(d: Any, dotkey: str) -> Any:
    keys = dotkey.split(".")
    cur = d
    for key in keys:
        if not hasattr(cur, "get"):
            raise ValueError(f"Try to load '{dotkey}' from `{d}`, but not found.")
        cur = cur.get(key, None)
        if cur is None:
            return None
    return cur


def create_dict_from_dotkey(dotkey: str, value: Any) -> Dict[str, Any]:
    keys = dotkey.split(".")
    result: Dict[str, Any] = {}
    cur = result
    for key in keys[:-1]:
        ncur = {}
        cur[key] = ncur
        cur = ncur
    cur[keys[-1]] = value
    return result


def merge_dicts(dict_a: Dict[Any, Any], dict_b: Dict[Any, Any]) -> Dict[Any, Any]:
    """dict_b has precedens

    >>> a = {1: 2, "a": {"x": 1, "y": {"z": 3}}}
    >>> b = {1: 3, "a": {"x": 2, "y": 10}}
    >>> expected = {1: 3, "a": {"x": 2, "y": 10}}
    """
    keys = set(dict_a.keys()) | set(dict_b.keys())
    ret: Dict[Any, Any] = {}
    for k in keys:
        if k in dict_a and k not in dict_b:
            ret[k] = dict_a[k]
        elif k not in dict_a and k in dict_b:
            ret[k] = dict_b[k]
        elif k in dict_a and k not in dict_b:
            va = dict_a[k]
            vb = dict_b[k]
            if isinstance(va, dict) and isinstance(vb, dict):
                ret[k] = merge_dicts(va, vb)  # type: ignore
            else:
                ret[k] = vb
        else:
            raise ValueError("Unreachable")
    return ret


def resolve_alias(aliases: Dict[str, str], cfg: Dict[str, Any]) -> Dict[str, Any]:
    for alias, name in aliases.items():
        v = get_by_dotkey(cfg, alias)
        if v is None:
            continue
        cfg = merge_dicts(cfg, create_dict_from_dotkey(name, v))
    return cfg


def yaml_to_dict(yaml_string: str) -> Dict[Any, Any]:
    try:
        ret = yaml.safe_load(yaml_string)
    except Exception as e:
        raise ValueError(f"Failed to parse yaml string: {yaml_string}") from e
    if not isinstance(ret, dict):
        raise ValueError(f"Not dictionary format: {yaml_string}")
    return ret  # type: ignore


"""
The following `SerializationMixin` is adopted from spacy-transformers,
which is distributed under the following license:

MIT License

Copyright (c) 2019 ExplosionAI GmbH

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


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

    serialization_fields: List[str] = []
    name: str

    def from_bytes(self, bytes_data: bytes, **kwargs: Any):
        pkls = srsly.pickle_loads(bytes_data)
        for field in self.serialization_fields:
            setattr(self, field, pkls[field])
        return self

    def to_bytes(self, **kwargs: Any):
        pkls = OrderedDict()
        for field in self.serialization_fields:
            pkls[field] = getattr(self, field, None)
        return srsly.pickle_dumps(pkls)

    def from_disk(self, path: Path, **kwargs: Any):
        path.mkdir(exist_ok=True)
        with (path / "data.pkl").open("rb") as file_:
            data = file_.read()
        return self.from_bytes(data, **kwargs)

    def to_disk(self, path: Path, **kwargs: Any):
        path.mkdir(exist_ok=True)
        data = self.to_bytes(**kwargs)
        with (path / "data.pkl").open("wb") as file_:
            file_.write(data)

    def require_model(self):
        if getattr(self, "model", None) in (None, True, False):
            raise ValueError(Errors.E109.format(name=self.name))


T = TypeVar("T")


def _setdefault(obj: Any, k: str, v: T) -> T:
    """Set attribute to object like dict.setdefault."""
    if hasattr(obj, k):
        return getattr(obj, k)
    setattr(obj, k, v)
    return v


def setdefaults(obj: Any, kv: Dict[str, Any]):
    """Set all attribute in kv to object"""
    for k, v in kv.items():
        _setdefault(obj, k, v)


def get_juman_command() -> Optional[Literal["juman", "jumanpp"]]:
    for cmd in ["jumanpp", "juman"]:
        if distutils.spawn.find_executable(cmd):
            return cmd  # type: ignore
    return None
