"""Convert KNP dependency parsing result to spacy format."""
from typing import Iterable, Optional

import spacy
from spacy.tokens import Doc, Span, Token
from spacy.symbols import ADJ, ADP, ADV, AUX, CCONJ, DET, NOUN, NUM, PRON, PART, PUNCT, VERB  # type: ignore

from camphr.pipelines.knp import KNP_USER_KEYS


@spacy.component("knp_dependency_parser", requires=("doc._.knp_tag_parent",))
def knp_dependency_parser(doc: Doc) -> Doc:
    tag_spans: Iterable[Span] = doc._.get(KNP_USER_KEYS.tag.spans)
    for tag in tag_spans:
        parent: Optional[Span] = tag._.get(KNP_USER_KEYS.tag.parent)
        if parent is not None:
            tag[0].head = parent[0]
            tag[0].dep_ = knp_dependency_parser_head(tag[0])
        else:
            tag[0].head = tag[0]
            tag[0].dep_ = "ROOT"
        for c in tag[1:]:
            c.head = tag[0]
            c.dep_ = knp_dependency_parser_func(c)
    doc.is_parsed = True
    return doc


def knp_dependency_parser_factory(*args, **kwargs):
    return knp_dependency_parser


def knp_dependency_parser_head(tag: Token) -> str:
    x = {ADV: "advmod", CCONJ: "advmod", DET: "det"}
    if tag.pos in x:  # type: ignore
        return x[tag.pos]  # type: ignore
    elif tag.pos in [VERB, ADJ]:  # type: ignore
        try:
            f = tag._.knp_morph_tag._.knp_tag_element.features
            if f["係"] == "連格":
                return "acl"
        except (AttributeError, KeyError):
            pass
        return "advcl"
    try:
        f = tag._.knp_morph_tag._.knp_tag_element.features
        k = f["係"] if f["係"] != "未格" else f["解析格"] + "格"
    except (AttributeError, KeyError):
        return "dep"
    x = {"隣": "nmod", "文節内": "compound", "ガ格": "nsubj", "ヲ格": "obj"}
    if k in x:
        return x[k]
    elif k != "ノ格":
        return "obl"
    if tag.head.pos in [VERB, ADJ]:  # type: ignore
        return "nsubj"
    elif tag.pos in [DET, PRON]:  # type: ignore
        tag.pos = DET  # type: ignore
        return "det"
    else:
        return "nmod"


def knp_dependency_parser_func(tag: Token) -> str:
    if tag.pos == AUX:  # type: ignore
        return "aux" if tag.head.pos in [VERB, ADJ] else "cop"  # type: ignore
    elif tag.pos == ADP:  # type: ignore
        return "mark" if tag.head.pos in [VERB, ADJ] else "case"  # type: ignore
    elif tag.pos == VERB:  # type: ignore
        if tag.head.pos == NOUN:  # type: ignore
            tag.head.pos = VERB  # type: ignore
        tag.pos = AUX  # type: ignore
        return "aux"
    elif tag.pos == PART:  # type: ignore
        return "mark"
    elif tag.pos == PUNCT:  # type: ignore
        return "punct"
    else:
        return "clf" if tag.head.pos == NUM else "flat"  # type: ignore
