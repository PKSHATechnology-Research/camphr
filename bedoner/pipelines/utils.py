import copy
import warnings
from enum import Enum
from typing import List, Tuple, Iterable
from spacy.tokens import Span, Doc
from spacy.gold import iob_to_biluo


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
    """Merge two ents. If ents1 is prior to ents0"""
    lents0 = sorted(ents0, key=lambda span: span.start)
    lents1 = sorted(ents1, key=lambda span: span.start)
    if len(lents0) == 0:
        return lents1
    if len(lents1) == 0:
        return lents0

    new_ents0 = []
    cur = 0
    left = 0
    right = lents1[cur].start

    for ent0 in lents0:
        start, end = ent0.start, ent0.end
        while True:
            if start >= left and end <= right:
                new_ents0.append(ent0)
                break
            elif start > right:
                left = lents1[cur].end
                cur += 1
                if len(lents1) <= cur:
                    right = len(ent0.doc) + 100
                else:
                    right = lents1[cur].start
            else:
                break

    return lents1 + new_ents0


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
