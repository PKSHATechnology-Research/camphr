import copy
import warnings
from enum import Enum
from itertools import chain
from typing import Callable, Iterable, List, Sequence, Tuple, TypeVar, Union

import numpy as np
from bedoner.errors import Warnings
from spacy.gold import iob_to_biluo
from spacy.tokens import Doc, Span, Token
from spacy.util import filter_spans


class BILUO(Enum):
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


def biluo_type(tag: str) -> BILUO:
    if tag.startswith("B-"):
        return B
    elif tag.startswith("I-"):
        return I
    elif tag.startswith("L-"):
        return L
    elif tag.startswith("U-"):
        return U
    elif tag == "O":
        return O
    return UNK


def deconstruct_biluo_tag(tag: str) -> Tuple[BILUO, str]:
    """Deconstruct string tag into BILUO type and its body"""
    biluo = biluo_type(tag)
    if biluo == UNK:
        return biluo, ""
    if biluo == O:
        return biluo, ""
    return biluo, tag[2:]


def is_group(tagl: BILUO, bl: str, tagr: BILUO, br: str) -> bool:
    if bl != br:
        return False
    return tagl in {B, I} and tagr in {I, L}


def construct_biluo_tag(biluo: BILUO, body: str = "") -> str:
    if body:
        assert biluo not in {O, UNK}
        return biluo.value + "-" + body
    assert biluo in {O, UNK}
    return biluo.value


def bio_to_biluo(tags: List[str]) -> List[str]:
    warnings.warn(f"Use spacy.gold.iob_to_biluo instead", DeprecationWarning)
    return iob_to_biluo(tags)


def biluo_to_bio(tags: List[str]) -> List[str]:
    """convert biluo tags to bio tags. Input `tags` is expected to be syntactically correct."""
    tags = copy.copy(tags)
    for i, tag in enumerate(tags):
        t, b = deconstruct_biluo_tag(tag)
        if t == L:
            tags[i] = construct_biluo_tag(I, b)
        elif t == U:
            tags[i] = construct_biluo_tag(B, b)
    return tags


def correct_biluo_tags(tags: List[str]) -> List[str]:
    """Check and correct biluo tags list so that it can be assigned to `spacy.gold.spans_from_biluo_tags`.

    All invalid tags will be replaced with `-`
    """
    tags = ["O"] + copy.copy(tags) + ["O"]
    for i in range(len(tags) - 1):
        tagl = tags[i]
        tagr = tags[i + 1]

        tl, bl = deconstruct_biluo_tag(tagl)
        tr, br = deconstruct_biluo_tag(tagr)
        if tl == UNK:
            tags[i] == UNK.value
        if tr == UNK:
            tags[i + 1] == UNK.value

        # left check
        if tl in {B, I} and not ((tr == I or tr == L) and bl == br):
            # invalid pattern
            tags[i] = UNK.value

        # right check
        if tr in {I, L} and not ((tl == B or tl == I) and bl == br):
            # invalid pattern
            tags[i + 1] = UNK.value
    return tags[1:-1]


def correct_bio_tags(tags: List[str]) -> List[str]:
    """Check and correct bio tags list.

    All invalid tags will be replaced with `-`
    """
    tags = ["O"] + copy.copy(tags)
    for i in range(len(tags) - 1):
        tagl = tags[i]
        tagr = tags[i + 1]

        tl, bl = deconstruct_biluo_tag(tagl)
        tr, br = deconstruct_biluo_tag(tagr)
        if tl == UNK:
            tags[i] == UNK.value
        if tr == UNK:
            tags[i + 1] == UNK.value

        # right check
        if tr == I and not (tl in {B, I} and bl == br):
            # invalid pattern
            tags[i + 1] = UNK.value
    return tags[1:]


def merge_entities(ents0: Iterable[Span], ents1: Iterable[Span]) -> List[Span]:
    """Merge two ents. ents1 is prior to ents0"""
    Warnings.W0("merge_entities", "spacy.util.filter_spans")
    return filter_spans(list(ents1) + list(ents0))


def set_heads(doc: Doc, heads: List[int]) -> Doc:
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
        return self.cfg.setdefault(USER_HOOKS, {})

    def add_user_hook(self, k: str, fn: Callable):
        hooks = self.user_hooks
        hooks[k] = fn


def flatten_docs_to_sents(docs: Iterable[Doc]) -> List[Span]:
    return list(chain.from_iterable(list(doc.sents) for doc in docs))


T = TypeVar("T")


def chunk(seq: Sequence[T], nums: Sequence[int]) -> List[List[T]]:
    i = 0
    output = []
    for n in nums:
        j = i + n
        output.append(seq[i:j])
        i = j
    return output
