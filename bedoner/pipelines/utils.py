import copy
from enum import Enum
from typing import List, Tuple


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
    """convert bio tags to biluo tags. Input `tags` is expected to be syntactically correct."""
    tags = copy.copy(tags) + ["O"]
    for i in range(len(tags) - 1):
        tl, bl = deconstruct_biluo_tag(tags[i])
        tr, br = deconstruct_biluo_tag(tags[i + 1])

        # unit B to U
        if tl == B and not is_group(tl, bl, tr, br):
            tags[i] = construct_biluo_tag(U, bl)
        # I-{other tag} to L-
        if tl == I and not is_group(tl, bl, tr, br):
            tags[i] = construct_biluo_tag(L, bl)

    return tags[:-1]


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
