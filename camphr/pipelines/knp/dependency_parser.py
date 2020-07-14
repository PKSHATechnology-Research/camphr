"""Convert KNP dependency parsing result to spacy format."""
from typing import Any, Dict, Iterable, Optional, List

import spacy
from spacy.symbols import (
    ADJ,
    ADP,
    ADV,
    AUX,
    CCONJ,
    DET,
    NOUN,
    NUM,
    PART,
    PRON,
    PUNCT,
    VERB,
)
from spacy.tokens import Doc, Span, Token

from camphr.pipelines.knp import KNP_USER_KEYS


@spacy.component("knp_dependency_parser", requires=("doc._.knp_tag_parent",))
def knp_dependency_parser(doc: Doc) -> Doc:
    tag_spans: Iterable[Span] = doc._.get(KNP_USER_KEYS.tag.spans)
    s = []
    for tag in tag_spans:
        for c in tag[1:]:
            c.head = tag[0]
            c.dep_ = _get_child_dep(c)
        parent: Optional[Span] = tag._.get(KNP_USER_KEYS.tag.parent)
        if parent is not None:
            tag[0].head = parent[0]
            tag[0].dep_ = _get_dep(tag[0])
        else:
            tag[0].head = tag[0]
            tag[0].dep_ = "ROOT"
        s.append(tag[0])
    s = _modify_head_punct(s)
    s = _modify_head_flat(s)
    s = _modify_head_conj(s)
    doc.is_parsed = True
    return doc


def knp_dependency_parser_factory(*args, **kwargs):
    return knp_dependency_parser


def _get_dep(tag: Token) -> str:
    ret = ({ADV: "advmod", CCONJ: "advmod", DET: "det"}).get(tag.pos)
    if ret:
        return ret
    elif tag.pos in {VERB, ADJ}:
        if tag._.knp_morph_tag._.knp_tag_element.features.get("係", "") == "連格":
            return "acl"
        return "advcl"
    return _get_dep_noun(tag)


def _get_dep_noun(tag: Token) -> str:
    f: Dict[str, Any] = tag._.knp_morph_tag._.knp_tag_element.features
    if "係" not in f:
        return "dep"
    k = f["係"] if f["係"] != "未格" or "解析格" not in f else f["解析格"] + "格"
    x = {"隣": "nmod", "文節内": "compound", "ガ格": "nsubj", "ヲ格": "obj", "ガ２格": "dislocated"}
    if k in x:
        return x[k]
    elif k == "ノ格":
        if tag.head.pos in {VERB, ADJ}:
            return "nsubj"
        elif tag.pos in {DET, PRON}:
            tag.pos = DET
            return "det"
        else:
            return "nummod" if tag.pos == NUM else "nmod"
    elif "並列タイプ" in f:
        if tag.head.pos in {VERB, ADJ}:
            return "obl"
        else:
            return "conj"
    return "obl"


def _get_child_dep(tag: Token) -> str:
    p, pp = tag.pos, tag.head.pos
    if p == AUX:
        return "aux" if pp in {VERB, ADJ} else "cop"
    elif p == ADP:
        return "mark" if pp in {VERB, ADJ} else "case"
    elif p in {VERB, ADJ}:
        if pp == NOUN:
            tag.head.pos = VERB
        tag.pos = AUX
        return "aux"
    elif p == PART:
        return "mark"
    elif p == PUNCT:
        return "punct"
    else:
        return "clf" if pp == NUM else "flat"


def _modify_head_punct(heads: List[Token]) -> List[Token]:
    s = [t for t in heads]
    for i, t in enumerate(s):
        if t.pos != PUNCT:
            continue
        x = [u for u in t.rights]  # type: ignore
        if len(x) == 0:
            continue
        h = x[0]
        h.head = t.head
        h.dep_ = t.dep_
        x = x[1:] + [u for u in t.lefts]  # type: ignore
        x += [t, h] if h.dep_ == "ROOT" else [t]
        x += [u for u in s if u.head == t]
        for u in x:
            u.head = h
        t.dep_ = "punct"
        s[i] = h
    return s


def _modify_head_flat(heads: List[Token]) -> List[Token]:
    s = [t for t in heads]
    for i, t in enumerate(s):
        if not t.tag_.startswith("接頭辞"):
            continue
        x = [u for u in t.rights]  # type: ignore
        if len(x) == 0:
            continue
        h = x[0]
        if t.pos == NOUN and h.dep_ == "flat":
            d = "compound"
        elif t.pos == ADV and h.dep_ == "aux":
            d = "advmod"
            h.pos = VERB
        elif t.pos == PART and h.dep_ == "aux":
            d = "advmod"
            h.pos = ADJ
        else:
            continue
        h.head = t.head
        h.dep_ = t.dep_
        x = x[1:] + [u for u in t.lefts]  # type: ignore
        x += [t, h] if h.dep_ == "ROOT" else [t]
        x += [u for u in s if u.head == t]
        for u in x:
            u.head = h
        t.dep_ = d
        s[i] = h
    return s


def _modify_head_conj(heads: List[Token]) -> List[Token]:
    s = [t for t in heads]
    for t in s:
        while t.dep_ == "conj" and t.i < t.head.i:
            h = t.head
            t.head = h.head
            t.dep_ = h.dep_
            x = [h, t] if t.dep_ == "ROOT" else [h]
            x += [u for u in s if u.head == h and u.i < t.i]
            for u in x:
                u.head = t
            h.dep_ = "conj"
    return s
