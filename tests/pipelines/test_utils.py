import random
from itertools import zip_longest
from typing import List, Tuple

import pytest
import spacy
from bedoner import __version__
from bedoner.pipelines.utils import (
    B,
    I,
    O,
    UserHooksMixin,
    biluo_to_bio,
    bio_to_biluo,
    chunk,
    construct_biluo_tag,
    correct_biluo_tags,
    correct_bio_tags,
    flatten_docs_to_sents,
    merge_entities,
)
from hypothesis import given
from hypothesis import strategies as st
from spacy.gold import spans_from_biluo_tags
from spacy.language import Language
from spacy.pipeline import Sentencizer
from spacy.tests.util import get_doc
from spacy.tokens import Span
from spacy.vocab import Vocab


@pytest.fixture
def nlp():
    return Language()


def create_tags_sample(length=10, tags=["LOC", "PERSON", "DATE"]) -> List[str]:
    """Create completely random biluo tags. The output tags may be syntactically incorrect."""
    res = []
    for _ in range(length):
        pref = random.choice("BILUO")
        if pref == "O":
            res.append(pref)
            continue
        body = random.choice(tags)
        res.append(pref + "-" + body)
    return res


TESTCAESES = [(["B-LOC", "L-LOC", "O", "L-PERSON"], ["B-LOC", "L-LOC", "O", "-"])]


@pytest.mark.parametrize("tags,corrected", TESTCAESES)
def test_correct_biluo_tags_cases(nlp, tags, corrected):
    assert correct_biluo_tags(tags) == corrected


def test_correct_biluo_tags_random(nlp):
    ntests = 100
    for _ in range(ntests):
        length = 10
        text = ("foo " * length).strip()
        doc = nlp(text)
        tags = create_tags_sample(10)
        corrected_tags = correct_biluo_tags(tags)
        spans_from_biluo_tags(doc, corrected_tags)


def create_bio_tags_sample(length=10, tags=["LOC", "PERSON", "DATE"]) -> List[str]:
    """Create syntactically correct bio tag"""
    res = []
    prevb = ""
    for _ in range(length):
        b = random.choice(tags + [""])
        if b:
            if b != prevb:
                res.append(construct_biluo_tag(B, b))
            else:
                t = random.choice([I, B])
                res.append(construct_biluo_tag(t, b))
        else:
            res.append(construct_biluo_tag(O))
        prevb = b

    return res


@pytest.mark.xfail(__version__ >= "v0.8", reason="Deprecate bio_to_biluo", strict=True)
def test_bio_to_biluo(recwarn):
    bio_to_biluo(["B-PERSON", "I-PERSON", "O", "B-DATE"])


TESTCAESES_BILUO_BIO = [
    (["B-LOC", "L-LOC", "O", "U-PERSON"], ["B-LOC", "I-LOC", "O", "B-PERSON"])
]


@pytest.mark.parametrize("biluo,bio", TESTCAESES_BILUO_BIO)
def test_biluo_to_bio(biluo, bio):
    converted = biluo_to_bio(biluo)
    assert bio == converted


TESTCAESES_BIO = [(["B-LOC", "I-LOC", "O", "I-PERSON"], ["B-LOC", "I-LOC", "O", "-"])]


@pytest.mark.parametrize("tags,expected", TESTCAESES_BIO)
def test_correct_bio(tags, expected):
    corrected = correct_bio_tags(tags)
    assert corrected == expected


TESTCASES_MERGE_ENTS = [
    (
        ["This", "is", "an", "example", "text", "case"],
        [(0, 1, "A"), (2, 4, "B"), (5, 6, "E")],
        [(2, 3, "C"), (4, 5, "D")],
        [(0, 1, "A"), (2, 3, "C"), (4, 5, "D"), (5, 6, "E")],
    ),
    (
        ["This", "is", "an", "example", "text", "case"],
        [(0, 2, "A"), (4, 6, "E")],
        [(2, 3, "C"), (4, 5, "D")],
        [(0, 2, "A"), (2, 3, "C"), (4, 5, "D")],
    ),
    (
        ["This", "is", "an", "example", "text", "case"],
        [(0, 2, "A"), (5, 6, "E")],
        [(2, 3, "C"), (4, 5, "D")],
        [(0, 2, "A"), (2, 3, "C"), (4, 5, "D"), (5, 6, "E")],
    ),
    (
        ["This", "is", "an", "example", "text", "case", "foo", "bar", "a", "b", "c"],
        [(0, 3, "A"), (4, 5, "E"), (7, 9, "F")],
        [(2, 4, "C"), (5, 7, "D")],
        [(2, 4, "C"), (4, 5, "E"), (5, 7, "D"), (7, 9, "F")],
    ),
    (
        ["This", "is", "an", "example", "text", "case", "foo", "bar", "a", "b", "c"],
        [(0, 1, "A"), (4, 6, "E"), (6, 9, "F")],
        [(2, 4, "C"), (5, 7, "D")],
        [(0, 1, "A"), (2, 4, "C"), (5, 7, "D")],
    ),
]


def make_ents(doc, spans: List[Tuple[int, int, str]]) -> List[Span]:
    return [Span(doc, span[0], span[1], label=span[2]) for span in spans]


@pytest.mark.parametrize("words,spans0,spans1,expected_spans", TESTCASES_MERGE_ENTS)
def test_merge_ents(words, spans0, spans1, expected_spans):
    doc = get_doc(Vocab(), words=words)
    ents0 = make_ents(doc, spans0)
    ents1 = make_ents(doc, spans1)
    expected = make_ents(doc, expected_spans)

    merged = sorted(merge_entities(ents0, ents1), key=lambda span: span.start)
    for a, b in zip_longest(expected, merged):
        assert a.label_ == b.label_
        assert a.start == b.start
        assert a.end == b.end


class DummyForUserHooks(UserHooksMixin):
    def __init__(self):
        self.cfg = {}


def test_user_hooks_mixin():
    obj = DummyForUserHooks()
    obj.add_user_hook("foo", lambda x: 2 * x)
    assert obj.user_hooks["foo"](1) == 2


def test_flatten_docs_to_sens(vocab):
    sentencizer = Sentencizer(".")
    nlp = spacy.blank("en")
    nlp.add_pipe(sentencizer)
    texts = ["Foo is bar. Bar is baz.", "It is a sentence."]
    docs = nlp.pipe(texts)
    all_sents = flatten_docs_to_sents(docs)
    assert len(all_sents) == 3


@given(st.integers(1, 100), st.integers(1, 100), st.integers(-1000, 1000))
def test_chunk(l, max_num, seed):
    random.seed(seed)
    nums = [random.randint(1, max_num) for _ in range(l)]
    s = sum(nums)
    seq = [random.randint(-100, 100) for _ in range(s)]

    chunked = chunk(seq, nums)
    assert len(chunked) == len(nums)
    i = 0
    for n, a in zip(nums, chunked):
        j = i + n
        assert seq[i:j] == a
        i = j
