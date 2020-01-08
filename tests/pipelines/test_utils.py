import random
from itertools import product
from typing import List, Tuple

import pytest
import spacy
import torch
from camphr import __version__
from camphr.pipelines.utils import (
    EPS,
    B,
    I,
    O,
    UserHooksMixin,
    beamsearch,
    biluo_to_bio,
    bio_to_biluo,
    chunk,
    construct_biluo_tag,
    correct_biluo_tags,
    correct_bio_tags,
    flatten_docs_to_sents,
    minmax_scale,
)
from hypothesis import given
from hypothesis import strategies as st
from spacy.gold import spans_from_biluo_tags
from spacy.language import Language
from spacy.pipeline import Sentencizer
from spacy.tokens import Span


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
    assert correct_biluo_tags(tags) == (corrected, False)


def test_correct_biluo_tags_random(nlp):
    ntests = 100
    for _ in range(ntests):
        length = 10
        text = ("foo " * length).strip()
        doc = nlp(text)
        tags = create_tags_sample(10)
        corrected_tags, _ = correct_biluo_tags(tags)
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


class DummyForUserHooks(UserHooksMixin):
    def __init__(self):
        self.cfg = {}


def test_user_hooks_mixin():
    obj = DummyForUserHooks()
    obj.add_user_hook("foo", lambda x: 2 * x)
    assert obj.user_hooks["foo"](1) == 2


st_int = st.integers(1, 100)


@given(st_int, st_int, st_int, st.integers(-1000, 1000))
def test_beamsearch(n, m, k, s):
    torch.manual_seed(s)
    data = torch.randn(n, m)
    output = beamsearch(data, k)
    assert output.shape == (min(m ** n, k), n)
    assert all(output[0] == data.argmax(1))


def _elephant_beamsearch(data: torch.Tensor, k: int) -> torch.Tensor:
    data = minmax_scale(data) + EPS
    data_with_idx = [[(a, i) for i, a in enumerate(row)] for row in data]
    ents = []
    for items in product(*data_with_idx):
        score, seq = 1, []
        for s, i in items:
            score *= s
            seq.append(i)
        ents.append((score, seq))
    res = sorted(ents, reverse=True)[:k]
    return torch.tensor(list(zip(*res))[1])


st_small_int = st.integers(1, 4)


@given(st_small_int, st_small_int, st.integers(1, 4 ** 4), st.integers(-1000, 1000))
def test_with_elephant_beamsearch(n, m, k, s):
    torch.manual_seed(s)
    data = torch.randn(n, m)
    output = beamsearch(data, k)
    elephant_output = _elephant_beamsearch(data, k)
    assert torch.all(output == elephant_output)


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
