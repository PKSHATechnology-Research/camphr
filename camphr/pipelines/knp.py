"""Defines KNP pipelines."""
import re
from typing import Callable, Dict, Iterable, List, NamedTuple, Optional, Tuple

import spacy
from spacy.tokens import Doc, Span, Token
from spacy.util import filter_spans
from toolz import curry

from camphr.consts import JUMAN_LINES
from camphr.utils import get_juman_command

LOC2IOB = {"B": "B", "I": "I", "E": "I", "S": "B"}
Span.set_extension(JUMAN_LINES, default=None)
SKIP_TOKENS = {"@"}


TAG = "tag"
BUNSETSU = "bunsetsu"
MORPH = "morph"


class KnpUserKeyType(NamedTuple):
    element: str
    spans: str
    list_: str
    parent: str
    children: str


class KnpUserKeys(NamedTuple):
    tag: KnpUserKeyType
    bunsetsu: KnpUserKeyType
    morph: KnpUserKeyType


KNP_USER_KEYS = KnpUserKeys(
    *[
        KnpUserKeyType(
            *["knp_" + comp + f"_{type_}" for type_ in KnpUserKeyType._fields]
        )
        for comp in KnpUserKeys._fields
    ]
)


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


@spacy.component("knp", assigns=["doc.ents"])
class KNP:
    @staticmethod
    def install_extensions():
        K = KNP_USER_KEYS
        Token.set_extension(K.morph.element, default=None, force=True)
        for k in [
            K.bunsetsu.element,
            K.tag.element,
            K.bunsetsu.list_,
            K.morph.list_,
            K.tag.list_,
        ]:
            Span.set_extension(k, default=None, force=True)
        for k in [BUNSETSU, TAG]:
            Span.set_extension(getattr(KNP_USER_KEYS, k).spans, getter=get_knp_span(k))
            Span.set_extension(
                getattr(KNP_USER_KEYS, k).parent, getter=get_knp_parent(k)
            )
            Span.set_extension(
                getattr(KNP_USER_KEYS, k).children, getter=get_knp_children(k)
            )

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
            for l, comp in zip([blist, mlist, tlist], ["bunsetsu", "morph", "tag"]):
                sent._.set(getattr(KNP_USER_KEYS, comp).list_, l)
            assert len(mlist) == len(sent)
            for m, token in zip(mlist, sent):
                token._.set(KNP_USER_KEYS.morph.element, m)
        doc.ents = filter_spans(doc.ents + tuple(_extract_knp_ent(doc)))  # type: ignore
        # TODO: https://github.com/python/mypy/issues/3004
        return doc


@curry
def get_knp_span(type_: str, span: Span) -> List[Span]:
    """Get knp tag or bunsetsu list"""
    assert type_ != MORPH

    knp_list = span.sent._.get(getattr(KNP_USER_KEYS, type_).list_)
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


def get_knp_element_id(elem) -> int:
    from pyknp import Morpheme, Bunsetsu, Tag

    for cls, attr in [(Morpheme, "mrph_id"), (Bunsetsu, "bnst_id"), (Tag, "tag_id")]:
        if isinstance(elem, cls):
            return getattr(elem, attr)
    raise ValueError(type(elem))


@curry
def get_knp_parent(type_: str, span: Span) -> Optional[Span]:
    tag_or_bunsetsu = span._.get(getattr(KNP_USER_KEYS, type_).element)
    if not tag_or_bunsetsu:
        return None
    p = tag_or_bunsetsu.parent
    if not p:
        return None
    spans = span.sent._.get(getattr(KNP_USER_KEYS, type_).spans)
    return spans[get_knp_element_id(p)]


@curry
def get_knp_children(type_: str, span: Span) -> List[Span]:
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
            r"<NE:(\w+):(.*)?>", token._.get(KNP_USER_KEYS.morph.element).fstring
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


KNP.install_extensions()
