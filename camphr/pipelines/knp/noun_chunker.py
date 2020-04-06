from typing import Iterable, List, Optional, Tuple

from spacy.tokens import Doc, Span, Token

from .consts import KNP_USER_KEYS


def knp_noun_chunker(doc: Doc) -> Iterable[Tuple[int, int, str]]:
    ret = []
    for taglist in _extract_noun_phrases(doc):
        last = _extract_content(taglist[-1])
        ret.append((taglist[0].start, last.end, "NP"))
    return ret


def _spans_to_span_without_last_aux(spans: List[Span], label: str) -> Span:
    return _spans_to_span(spans[:-1] + [_extract_content(spans[1])], label)


def _spans_to_span(spans: List[Span], label: str) -> Span:
    return Span(spans[0].doc, spans[0].start, spans[-1].end, label=label)


def _extract_noun_phrases(doc: Doc) -> Iterable[List[Span]]:
    seen = [False for _ in range(len(doc))]
    ret = []
    for tag in reversed(list(doc._.get(KNP_USER_KEYS.tag.spans))):
        if tag._.get(KNP_USER_KEYS.tag.element).features.get("体言"):
            taglist = _traverse_children(tag)
            i, j = _get_span(taglist)
            if seen[i]:
                # It is sufficient to check if the start has already been seen
                # because we are traversing the tags from the end of the doc.
                continue
            for k in range(i, j):
                seen[k] = True
            ret.append(taglist)
    return reversed(ret)


def _is_content(token: Token) -> bool:
    return "内容語>" in token._.get(KNP_USER_KEYS.morph.element).fstring


def _extract_content(tag: Span) -> Span:
    start = None
    end = None
    for token in tag:
        if _is_content(token):
            if start is None:
                start = token.i
            end = token.i + 1
        else:
            break
    assert start is not None and end is not None
    return Span(tag.doc, start, end)


def _traverse_children(
    tag: Span, _is_root: bool = True, _root: Optional[Span] = None
) -> List[Span]:
    """Traverse children except for `para` dependency.
    
    Args:
        tag: tag to be traversed
        _is_root: internally used parameter. Should not be changed outside this function.
        _root: internally used parameter. Should not be changed outside this function.
    """
    _root = _root or tag
    if (
        not _is_root
        and tag._.get(KNP_USER_KEYS.tag.element).dpndtype == "P"
        and tag._.get(KNP_USER_KEYS.tag.parent) == _root
    ):
        return []
    result = []
    for c in tag._.get(KNP_USER_KEYS.tag.children):
        result.extend(_traverse_children(c, False, _root))
    result.append(tag)
    return result


def _get_span(taglist: List[Span]) -> Tuple[int, int]:
    i = taglist[0].start
    j = taglist[-1].end
    return i, j
