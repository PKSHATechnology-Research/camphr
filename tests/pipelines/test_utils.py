import random
from typing import List

import pytest
from bedoner.pipelines.utils import (
    B,
    I,
    O,
    bio_to_biluo,
    construct_biluo_tag,
    correct_biluo_tags,
)
from spacy.gold import spans_from_biluo_tags
from spacy.language import Language


@pytest.fixture
def nlp():
    return Language()


def create_tags_sample(length=10, tags=["LOC", "PERSON", "DATE"]) -> List[str]:
    """Create completely random biluo tags. The output tags may not be syntactically incorrect."""
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


def test_bio_to_biluo():
    ntests = 100
    for _ in range(ntests):
        length = 10
        tags = create_bio_tags_sample(length)
        biluo_tags = bio_to_biluo(tags)
        assert biluo_tags == correct_biluo_tags(biluo_tags), tags
