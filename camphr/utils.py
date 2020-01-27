"""The utils module defines util functions used accross sub packages."""
import bisect
import importlib
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, Union, cast

import spacy
import srsly
import yaml
from cytoolz import curry
from more_itertools import padded
from omegaconf import Config, OmegaConf
from spacy.errors import Errors
from spacy.language import BaseDefaults
from spacy.tokens import Doc, Span, Token
from spacy.util import filter_spans

from camphr.types import Pathlike
from camphr.VERSION import __version__


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

    serialization_fields = []

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

    def require_model(self):
        if getattr(self, "model", None) in (None, True, False):
            raise ValueError(Errors.E109.format(name=self.name))


def zero_pad(a: List[List[int]], pad_value: int = 0) -> List[List[int]]:
    """Padding the input so that the lengths of the inside lists are all equal."""
    if len(a) == 0:
        return []
    max_length = max(len(el) for el in a)
    if max_length == 0:
        return a
    return [list(padded(el, fillvalue=pad_value, n=max_length)) for el in a]


RE_URL = re.compile(
    r"https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)"
)


def token_from_char_pos(doc: Doc, i: int) -> Token:
    token_idxs = [t.idx for t in doc]
    return doc[bisect.bisect(token_idxs, i) - 1]


def destruct_token(doc: Doc, *char_pos: int) -> Doc:
    for i in char_pos:
        with doc.retokenize() as retokenizer:
            token = token_from_char_pos(doc, i)
            heads = [token] * len(token)
            retokenizer.split(doc[token.i], list(token.text), heads=heads)


def get_doc_char_span(
    doc: Doc, i: int, j: int, destructive: bool = True, **kwargs
) -> Optional[Span]:
    """Get Span from Doc with char position, similar to doc.char_span.

    Args:
        destructive: If True, tokens in [i,j) will be splitted and make sure to return span.
        kwargs: passed to Doc.char_span
    """
    span = doc.char_span(i, j, **kwargs)
    if not span and destructive:
        try:
            destruct_token(doc, i, j)
            span = doc.char_span(i, j, **kwargs)
        except AssertionError:
            # TODO: https://github.com/explosion/spaCy/issues/4604
            pass
    return span


def get_doc_char_spans_list(
    doc: Doc, spans: Iterable[Tuple[int, int]], destructive: bool = True, **kwargs
) -> List[Span]:
    res = []
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


def get_sents(doc: Doc) -> Iterable[Span]:
    if doc.is_sentenced:
        return doc.sents
    return doc[:]


def import_attr(import_path: str) -> Any:
    items = import_path.split(".")
    module_name = ".".join(items[:-1])
    return getattr(importlib.import_module(module_name), items[-1])


def get_requirements_line():
    # TODO: modify after release
    return f"camphr @ git+https://github.com/PKSHATechnology/bedore-ner@v0.5"


def get_defaults(lang: str) -> Type[BaseDefaults]:
    try:
        lang_cls = spacy.util.get_lang_class(lang)
    except Exception:
        return BaseDefaults
    return getattr(lang_cls, "Defaults", BaseDefaults)


def get_labels(labels_or_path: Union[List[str], Pathlike]) -> List[str]:
    if isinstance(labels_or_path, (str, Path)):
        path = Path(labels_or_path)
        if path.suffix == ".json":
            return srsly.read_json(labels_or_path)
        elif path.suffix in {".yml", ".yaml"}:
            return yaml.safe_load(path.read_text())
    return cast(List[str], labels_or_path)


def get_by_dotkey(d: dict, dotkey: str) -> Any:
    assert dotkey
    keys = dotkey.split(".")
    cur = d
    for key in keys:
        cur = cur.get(key, None)
        if cur is None:
            return None
    return cur


def create_dict_from_dotkey(dotkey: str, value: Any) -> Dict[str, Any]:
    assert dotkey
    keys = dotkey.split(".")
    result = {}
    cur = result
    for key in keys[:-1]:
        cur[key] = {}
        cur = cur[key]
    cur[keys[-1]] = value
    return result


@curry
def resolve_alias(aliases: Dict[str, str], cfg: Config) -> Config:
    for alias, name in aliases.items():
        v = get_by_dotkey(cfg, alias)
        if v is None:
            continue
        cfg = OmegaConf.merge(cfg, OmegaConf.create(create_dict_from_dotkey(name, v)))
    return cfg
