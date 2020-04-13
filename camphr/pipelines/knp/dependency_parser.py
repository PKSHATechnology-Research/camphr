"""Convert KNP dependency parsing result to spacy format."""
from typing import Any, Dict, Iterable, Optional

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
    for tag in tag_spans:
        parent: Optional[Span] = tag._.get(KNP_USER_KEYS.tag.parent)
        if parent is not None:
            tag[0].head = parent[0]
            tag[0].dep_ = _get_dep(tag[0])
        else:
            tag[0].head = tag[0]
            tag[0].dep_ = "ROOT"
        for c in tag[1:]:
            c.head = tag[0]
            c.dep_ = _get_child_dep(c)
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
    if "係" not in f and "解析格" not in f:
        return "dep"
    k = f["係"] if f["係"] != "未格" else f["解析格"] + "格"
    x = {"隣": "nmod", "文節内": "compound", "ガ格": "nsubj", "ヲ格": "obj"}
    if k in x:
        return x[k]
    elif k != "ノ格":
        return "obl"
    if tag.head.pos in {VERB, ADJ}:
        return "nsubj"
    elif tag.pos in {DET, PRON}:
        tag.pos = DET
        return "det"
    else:
        return "nmod"


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
