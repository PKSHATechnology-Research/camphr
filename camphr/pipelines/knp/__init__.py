"""Defines KNP pipelines."""
import functools
import re
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple

import spacy
from spacy.tokens import Doc, Span, Token
from spacy.util import filter_spans
from toolz import curry
from typing_extensions import Literal

from camphr.consts import JUMAN_LINES
from camphr.utils import get_juman_command

from .consts import KNP_USER_KEYS
from .noun_chunker import knp_noun_chunker

LOC2IOB = {"B": "B", "I": "I", "E": "I", "S": "B"}
Span.set_extension(JUMAN_LINES, default=None)
SKIP_TOKENS = {"@"}


TAG = "tag"
BUNSETSU = "bunsetsu"
MORPH = "morph"
L_KNP_OBJ = Literal["tag", "bunsetsu", "morph"]


def _take_juman_lines(n: int, juman_lines: List[str]) -> Tuple[List[str], List[str]]:
    lines = []
    count = 0
    for line in juman_lines:
        lines.append(line)
        head = line.split(" ")[0]
        if head not in SKIP_TOKENS:
            count += 1
        if count >= n:
            break
    return lines, juman_lines[len(lines) :]


@spacy.component("juman_sentencizer", requires=["doc.sents"])
def juman_sentencizer(doc: Doc) -> Doc:
    """Split juman_string for knp"""
    juman_lines = doc.user_data[JUMAN_LINES].strip("\n").split("\n")
    for sent in doc.sents:
        lines, juman_lines = _take_juman_lines(len(sent), juman_lines)
        sent._.set(JUMAN_LINES, "\n".join(lines) + "\n" + "EOS")
    return doc


# For spacy factory. see pyproject.toml
def juman_sentencizer_factory(*args, **kwargs):
    return juman_sentencizer


@spacy.component("knp", assigns=["doc.ents", "doc._.knp_tag_parent"])
class KNP:
    def __init__(
        self,
        knp_kwargs: Optional[Dict[str, str]] = None,
        preprocessor: Callable[[str], str] = None,
    ):
        import pyknp

        cmd = get_juman_command()
        assert cmd
        knp_kwargs = knp_kwargs or {}
        knp_kwargs.setdefault("jumancommand", cmd)

        self.knp = pyknp.KNP(**knp_kwargs)
        self.knp_kwargs = knp_kwargs

    @classmethod
    def from_nlp(cls, nlp, *args, **kwargs):
        return cls()

    def __call__(self, doc: Doc) -> Doc:
        for sent in doc.sents:
            blist = self.knp.parse_juman_result(sent._.get(JUMAN_LINES))
            mlist = blist.mrph_list()
            tlist = blist.tag_list()
            if len(mlist) != len(sent):
                mlist = _separate_mrph(mlist, sent)
            for label, comp in zip([blist, mlist, tlist], ["bunsetsu", "morph", "tag"]):
                sent._.set(getattr(KNP_USER_KEYS, comp).list_, label)
            for m, token in zip(mlist, sent):
                token._.set(KNP_USER_KEYS.morph.element, m)
        doc.ents = filter_spans(doc.ents + tuple(_extract_knp_ent(doc)))  # type: ignore
        doc.noun_chunks_iterator = knp_noun_chunker  # type: ignore
        # TODO: https://github.com/python/mypy/issues/3004
        return doc


def _separate_mrph(mlist: List, sent: Span) -> List:
    mm = []
    i = 0
    for m in mlist:
        if "形態素連結" in m.fstring:
            j = len(m.midasi)
            while j > 0:
                mm.append(m)
                j -= len(sent[i].text)
                i += 1
        elif sent[i].text == m.midasi:
            mm.append(m)
            i += 1
        else:
            raise ValueError(
                f"""Internal error occured
            Sentence: {sent.text}
            mlist : {[m.midasi for m in mlist]}
            tokens: {[t.text for t in sent]}
            diff  : {m.midasi}, {sent[i].text}
            """
            )
    return mm


@curry
@functools.lru_cache()
def token_to_knp_span(type_: str, token: Token) -> Span:
    """Returns the knp span containing the token."""
    assert type_ != MORPH
    for tag_or_bunsetsu in token.doc._.get(getattr(KNP_USER_KEYS, type_).spans):
        if token.i < tag_or_bunsetsu.end:
            return tag_or_bunsetsu
    raise ValueError("internal error")


@curry
@functools.lru_cache()
def get_knp_span(type_: str, span: Span) -> List[Span]:
    """Get knp tag or bunsetsu list"""
    assert type_ != MORPH

    # TODO: span._ => span.sent._
    knp_list = span._.get(getattr(KNP_USER_KEYS, type_).list_)
    if not knp_list:
        return []

    res = []
    i = span.start_char
    doc = span.doc
    for b in knp_list:
        j = i + len(b.midasi)
        bspan = doc.char_span(i, j)
        bspan._.set(getattr(KNP_USER_KEYS, type_).element, b)
        res.append(bspan)
        i = j
    return res


@functools.lru_cache()
def get_knp_element_id(elem) -> int:
    from pyknp import Morpheme, Bunsetsu, Tag

    for cls, attr in [(Morpheme, "mrph_id"), (Bunsetsu, "bnst_id"), (Tag, "tag_id")]:
        if isinstance(elem, cls):
            return getattr(elem, attr)
    raise ValueError(type(elem))


@curry
def get_all_knp_features_from_sents(
    type_: L_KNP_OBJ, feature: str, doc: Doc
) -> Iterator[Any]:
    """Helper for spacy.doc extension to get knp features from spans and concatenate them."""
    for sent in doc.sents:
        key = getattr(KNP_USER_KEYS, type_)
        yield from sent._.get(getattr(key, feature))


@curry
@functools.lru_cache()
def get_knp_parent(type_: L_KNP_OBJ, span: Span) -> Optional[Span]:
    tag_or_bunsetsu = span._.get(getattr(KNP_USER_KEYS, type_).element)
    if not tag_or_bunsetsu:
        return None
    p = tag_or_bunsetsu.parent
    if not p:
        return None
    spans = span.sent._.get(getattr(KNP_USER_KEYS, type_).spans)
    return spans[get_knp_element_id(p)]


@curry
@functools.lru_cache()
def get_knp_children(type_: L_KNP_OBJ, span: Span) -> List[Span]:
    tag_or_bunsetsu = span._.get(getattr(KNP_USER_KEYS, type_).element)
    if not tag_or_bunsetsu:
        return []
    children = tag_or_bunsetsu.children
    spans = span.sent._.get(getattr(KNP_USER_KEYS, type_).spans)
    return [spans[get_knp_element_id(child)] for child in children]


def _extract_knp_ent(doc: Doc) -> List[Span]:
    ents: List[Tuple[str, int, int]] = []
    for token in doc:
        ent_match = re.search(
            r"<NE:(\w+):(.*?)?>", token._.get(KNP_USER_KEYS.morph.element).fstring
        )
        if ent_match:
            ent, loc = ent_match.groups()
            iob = LOC2IOB[loc]
            if iob == "B":
                ents.append((ent, token.i, token.i + 1))
            else:
                last = ents[-1]
                ents[-1] = (last[0], last[1], token.i + 1)
    spacy_ents = _create_ents(doc, ents)
    return spacy_ents


def _create_ents(doc: Doc, ents: Iterable[Tuple[str, int, int]]) -> List[Span]:
    new_ents = []
    for text, start, end in ents:
        new_ents.append(Span(doc, start, end, label=text))
    return filter_spans(doc.ents + tuple(new_ents))


def _install_extensions():
    K = KNP_USER_KEYS
    Token.set_extension(K.morph.element, default=None, force=True)
    for k in ["bunsetsu", "tag"]:
        Token.set_extension(getattr(K.morph, k), getter=token_to_knp_span(k))
    for k in ["bunsetsu", "morph", "tag"]:
        for feature in ["element", "list_"]:
            key = getattr(getattr(K, k), feature)
            Span.set_extension(key, default=None, force=True)
    for k in ["bunsetsu", "morph", "tag"]:
        for feature in ["spans", "list_"]:
            key = getattr(getattr(K, k), feature)
            Doc.set_extension(key, getter=get_all_knp_features_from_sents(k, feature))
    for k in [BUNSETSU, TAG]:
        Span.set_extension(getattr(KNP_USER_KEYS, k).spans, getter=get_knp_span(k))
        Span.set_extension(getattr(KNP_USER_KEYS, k).parent, getter=get_knp_parent(k))
        Span.set_extension(
            getattr(KNP_USER_KEYS, k).children, getter=get_knp_children(k)
        )


_install_extensions()
