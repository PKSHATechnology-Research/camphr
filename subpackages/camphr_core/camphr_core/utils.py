"""The utils module defines util functions used accross sub packages."""
import bisect
import copy
import distutils.spawn
import importlib
from itertools import chain
import json
import operator
from pathlib import Path
import pickle
import re
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    TextIO,
    Tuple,
    TypeVar,
    Union,
    cast,
    overload,
)
from typing_extensions import Literal


from more_itertools import padded
import numpy as np

from camphr_core.doc import Doc, T_Span, T_Token, Token, Span

#  from spacy.tokens import Doc, Span, Token
import yaml

from camphr_core.VERSION import __version__


GoldCat = Dict[str, float]


def goldcat_to_label(goldcat: GoldCat) -> str:
    assert len(goldcat)
    return max(goldcat.items(), key=operator.itemgetter(1))[0]


def dump_jsonl(f: TextIO, dat: Iterable[Any]):
    for line in dat:
        f.write(json.dumps(line))


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


def token_from_char_pos(doc: Doc[T_Token, T_Span], i: int) -> T_Token:
    token_idxs = [t.idx for t in doc]
    return doc[bisect.bisect(token_idxs, i) - 1]


def _get_covering_span(
    doc: Doc[T_Token, T_Span], i: int, j: int, **kwargs: Any
) -> T_Span:
    token_idxs = [t.idx for t in doc]
    i = bisect.bisect(token_idxs, i) - 1
    j = bisect.bisect_left(token_idxs, j)
    span_cls = type(doc[:])
    return span_cls(doc, i, j, **kwargs)


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


def get_labels(labels_or_path: Union[List[str], Path]) -> List[str]:
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
        ncur: Dict[str, Any] = {}
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
        elif k in dict_a and k in dict_b:
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
        pkls = pickle.loads(bytes_data)
        for field in self.serialization_fields:
            setattr(self, field, pkls[field])
        return self

    def to_bytes(self, **kwargs: Any):
        pkls: Dict[str, Any] = dict()
        for field in self.serialization_fields:
            pkls[field] = getattr(self, field, None)
        return pickle.dumps(pkls)

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
            raise ValueError("self.model is not instantiated")


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


class BILUO:
    B = "B"
    I = "I"  # noqa: E741
    L = "L"
    U = "U"
    O = "O"  # noqa: E741
    UNKNOWN = "-"


B = BILUO.B
I = BILUO.I  # noqa: E741
L = BILUO.L
U = BILUO.U
O = BILUO.O  # noqa: E741
UNK = BILUO.UNKNOWN


def biluo_type(tag: str) -> str:
    for prefix, biluo in [("B-", B), ("I-", I), ("L-", L), ("U-", U)]:
        if tag.startswith(prefix):
            return biluo
    if tag == "O":
        return O
    return UNK


def deconstruct_biluo_label(label: str) -> Tuple[str, str]:
    """Deconstruct biluo label into BILUO prefix and its label type"""
    biluo = biluo_type(label)
    if biluo == UNK:
        return biluo, ""
    if biluo == O:
        return biluo, ""
    return biluo, label[2:]


def is_group(tagl: BILUO, bl: str, tagr: BILUO, br: str) -> bool:
    if bl != br:
        return False
    return tagl in {B, I} and tagr in {I, L}


def construct_biluo_tag(biluo: str, body: str = "") -> str:
    if body:
        assert biluo not in {O, UNK}
        return biluo + "-" + body
    assert biluo in {O, UNK}
    return biluo


def bio_to_biluo(tags: List[str]) -> List[str]:
    raise ValueError("Deprecated: Use spacy.gold.iob_to_biluo instead")


def biluo_to_bio(tags: List[str]) -> List[str]:
    """convert biluo tags to bio tags. Input `tags` is expected to be syntactically correct."""
    tags = copy.copy(tags)
    for i, tag in enumerate(tags):
        t, b = deconstruct_biluo_label(tag)
        if t == L:
            tags[i] = construct_biluo_tag(I, b)
        elif t == U:
            tags[i] = construct_biluo_tag(B, b)
    return tags


def correct_biluo_tags(tags: List[str]) -> Tuple[List[str], bool]:
    """Check and correct biluo tags list so that it can be assigned to `spacy.gold.spans_from_biluo_tags`.

    All invalid tags will be replaced with `-`
    """
    is_correct = True
    tags = ["O"] + copy.copy(tags) + ["O"]
    for i in range(len(tags) - 1):
        tagl = tags[i]
        tagr = tags[i + 1]

        type_l, body_l = deconstruct_biluo_label(tagl)
        type_r, body_r = deconstruct_biluo_label(tagr)

        # left check
        if type_l in {B, I} and not ((type_r == I or type_r == L) and body_l == body_r):
            is_correct = False
            tags[i] = UNK

        # right check
        if type_r in {I, L} and not ((type_l == B or type_l == I) and body_l == body_r):
            is_correct = False
            tags[i + 1] = UNK
    return tags[1:-1], is_correct


def correct_bio_tags(tags: List[str]) -> Tuple[List[str], bool]:
    """Check and correct bio tags list.

    All invalid tags will be replaced with `-`
    """
    tags = copy.copy(tags)
    is_correct = True
    for i, (tagl, tagr) in enumerate(zip(tags, tags[1:])):
        tl, bl = deconstruct_biluo_label(tagl)
        tr, br = deconstruct_biluo_label(tagr)
        # Convert invalid I to B
        if tr == I and not (tl in {B, I} and bl == br):
            tags[i + 1] = construct_biluo_tag(B, br)
    return tags, is_correct


@overload
def set_heads(doc: Span, heads: List[int]) -> Span:
    ...


@overload
def set_heads(doc: Doc, heads: List[int]) -> Doc:
    ...


def set_heads(doc, heads):
    """Set heads to doc in UD annotation style.

    If fail to set, return doc without doing anything.
    """
    if max(heads) > len(doc) or min(heads) < 0:
        return doc
    for head, token in zip(heads, doc):
        if head == 0:
            token.head = token
        else:
            token.head = doc[head - 1]
    return doc


def get_doc_vector_via_tensor(doc: Doc) -> np.ndarray:
    return doc.tensor.sum(0)


def get_span_vector_via_tensor(span: Span) -> np.ndarray:
    return span.doc.tensor[span.start : span.end].sum(0)


def get_token_vector_via_tensor(token: Token) -> np.ndarray:
    return token.doc.tensor[token.i]


def get_similarity(o1: Union[Doc, Span, Token], o2: Union[Doc, Span, Token]) -> float:
    v1: np.ndarray = o1.vector
    v2: np.ndarray = o2.vector
    return (v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))).item()


USER_HOOKS = "user_hooks"


class UserHooksMixin:
    @property
    def user_hooks(self):
        return self.cfg.setdefault(USER_HOOKS, {})  # type: ignore

    def add_user_hook(self, k: str, fn: Callable):
        hooks = self.user_hooks
        hooks[k] = fn


def flatten_docs_to_sents(docs: Iterable[Doc]) -> List[Span]:
    return list(chain.from_iterable(list(doc.sents) for doc in docs))


def chunk(seq: Sequence[T], nums: Sequence[int]) -> List[Sequence[T]]:
    i = 0
    output = []
    for n in nums:
        j = i + n
        output.append(seq[i:j])
        i = j
    return output
