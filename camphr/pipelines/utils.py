import copy
import warnings
from enum import Enum
from itertools import chain
from typing import Callable, Iterable, List, Sequence, Tuple, TypeVar, Union

import numpy as np
import torch
from spacy.gold import iob_to_biluo
from spacy.tokens import Doc, Span, Token
from spacy.util import filter_spans

from camphr.errors import Warnings


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


def correct_biluo_tags(tags: List[str]) -> Tuple[List[str], bool]:
    """Check and correct biluo tags list so that it can be assigned to `spacy.gold.spans_from_biluo_tags`.

    All invalid tags will be replaced with `-`
    """
    is_correct = True
    tags = ["O"] + copy.copy(tags) + ["O"]
    for i in range(len(tags) - 1):
        tagl = tags[i]
        tagr = tags[i + 1]

        type_l, body_l = deconstruct_biluo_tag(tagl)
        type_r, body_r = deconstruct_biluo_tag(tagr)
        if type_l == UNK:
            tags[i] == UNK.value
        if type_r == UNK:
            tags[i + 1] == UNK.value

        # left check
        if type_l in {B, I} and not ((type_r == I or type_r == L) and body_l == body_r):
            is_correct = False
            tags[i] = UNK.value

        # right check
        if type_r in {I, L} and not ((type_l == B or type_l == I) and body_l == body_r):
            is_correct = False
            tags[i + 1] = UNK.value
    return tags[1:-1], is_correct


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


def minmax_scale(data: torch.Tensor) -> torch.Tensor:
    M, m = data.max(), data.min()
    if M == m:
        return data / M
    return (data - m) / (M - m)


EPS = 1e-5


def beamsearch(data: torch.Tensor, k: int) -> torch.Tensor:
    """Beam search for sequential scores

    Args:
        data: tensor of shape (length, d). requires d > 0
        k: beam width
    Returns: (k, length) tensor"""
    assert len(data.shape) == 2
    if len(data) == 0:
        return torch.zeros(k, 0)

    # scaling for score
    data = minmax_scale(data) + EPS

    _, m = data.shape
    scores, candidates = torch.topk(data[0], k=min(k, m))
    candidates = candidates[:, None]

    for row in data[1:]:
        z = torch.einsum("i,j->ij", scores, row).flatten()
        scores, flat_idx = torch.topk(z, k=min(k, len(z)))
        i, j = flat_idx // m, flat_idx % m
        candidates = torch.cat([candidates[i], j[:, None]], dim=-1)
    return candidates


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
