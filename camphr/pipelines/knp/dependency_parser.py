"""Convert KNP dependency parsing result to spacy format."""
from typing import Iterable, Optional

import spacy
from spacy.tokens import Doc, Span
from spacy.symbols import ADJ, ADP, ADV, AUX, CCONJ, DET, NOUN, NUM, PRON, PART, PUNCT, VERB  # type: ignore

from camphr.pipelines.knp import KNP_USER_KEYS


@spacy.component("knp_dependency_parser", requires=("doc._.knp_tag_parent",))
def knp_dependency_parser(doc: Doc) -> Doc:
    tag_spans: Iterable[Span] = doc._.get(KNP_USER_KEYS.tag.spans)
    for tag in tag_spans:
        parent: Optional[Span] = tag._.get(KNP_USER_KEYS.tag.parent)
        if parent is not None:
            tag[0].head = parent[0]
            x = {ADV: "advmod", CCONJ: "advmod", DET: "det"}
            if tag[0].pos in x:  # type: ignore
                tag[0].dep_ = x[tag[0].pos]  # type: ignore
            elif tag[0].pos in [VERB, ADJ]:  # type: ignore
                tag[0].dep_ = "advcl"
                try:
                    f = tag[0]._.knp_morph_tag._.knp_tag_element.features
                    if f["係"] == "連格":
                        tag[0].dep_ = "acl"
                except (AttributeError, KeyError):
                    pass
            else:
                tag[0].dep_ = "dep"
                try:
                    f = tag[0]._.knp_morph_tag._.knp_tag_element.features
                    k = f["係"] if f["係"] != "未格" else f["解析格"] + "格"
                    x = {"隣": "nmod", "文節内": "compound", "ガ格": "nsubj", "ヲ格": "obj"}
                    if k in x:
                        tag[0].dep_ = x[k]
                    elif k == "ノ格":
                        if parent[0].pos in [VERB, ADJ]:  # type: ignore
                            tag[0].dep_ = "nsubj"
                        elif tag[0].pos in [DET, PRON]:  # type: ignore
                            tag[0].pos = DET  # type: ignore
                            tag[0].dep_ = "det"
                        else:
                            tag[0].dep_ = "nmod"
                    else:
                        tag[0].dep_ = "obl"
                except (AttributeError, KeyError):
                    pass
        else:
            tag[0].head = tag[0]
            tag[0].dep_ = "ROOT"
        for c in tag[1:]:
            c.head = tag[0]
            if c.pos == AUX:  # type: ignore
                c.dep_ = "aux" if tag[0].pos in [VERB, ADJ] else "cop"  # type: ignore
            elif c.pos == ADP:  # type: ignore
                c.dep_ = "mark" if tag[0].pos in [VERB, ADJ] else "case"  # type: ignore
            elif c.pos == VERB:  # type: ignore
                if tag[0].pos == NOUN:  # type: ignore
                    tag[0].pos = VERB  # type: ignore
                c.pos = AUX  # type: ignore
                c.dep_ = "aux"
            elif c.pos == PART:  # type: ignore
                c.dep_ = "mark"
            elif c.pos == PUNCT:  # type: ignore
                c.dep_ = "punct"
            else:
                c.dep_ = "clf" if tag[0].pos == NUM else "flat"  # type: ignore
    doc.is_parsed = True
    return doc


def knp_dependency_parser_factory(*args, **kwargs):
    return knp_dependency_parser
