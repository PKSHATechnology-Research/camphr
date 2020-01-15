from pathlib import Path
from typing import List, Tuple

import pytest
from camphr.models import create_model
from camphr.pipelines.knp import KNP_USER_KEYS
from omegaconf import OmegaConf
from spacy.language import Language
from spacy.tokens import Doc

from ..utils import check_knp

pytestmark = pytest.mark.skipif(not check_knp(), reason="knp is not always necessary")


@pytest.fixture
def nlp():
    cfgpath = Path(__file__).parent.parent.parent / "camphr/model_config" / "knp.yml"
    return create_model(OmegaConf.load(str(cfgpath.absolute())))


TEXTS = ["今日は\u3000いい天気だったので山田太郎と散歩に行きました。帰りni富士山が見えた。"]
TESTCASES = [(TEXTS[0], [("今日", "DATE"), ("山田太郎", "PERSON"), ("富士山", "LOCATION")])]


@pytest.mark.parametrize("text,ents", TESTCASES)
def test_knp_entity_extractor(nlp: Language, text: str, ents: Tuple[str]):
    doc: Doc = nlp(text)
    assert len(doc.ents) == len(ents)
    for s, expected_ent in zip(doc.ents, ents):
        assert s.text == expected_ent[0]
        assert s.label_ == expected_ent[1]


@pytest.mark.parametrize("text", TEXTS)
def test_knp_span_getter(nlp: Language, text: str):
    doc: Doc = nlp(text)
    for sent in doc.sents:
        blist = sent._.get(KNP_USER_KEYS.bunsetsu.list_)
        text = "".join(b.midasi for b in blist)
        assert text == sent.text
        assert all(
            [
                b.midasi == s.text
                for b, s in zip(blist, sent._.get(KNP_USER_KEYS.bunsetsu.spans))
            ]
        )
        assert all(
            [
                t.midasi == s.text
                for t, s in zip(blist.tag_list(), sent._.get(KNP_USER_KEYS.tag.spans))
            ]
        )


@pytest.mark.parametrize(
    "text,parents",
    [
        (
            "太郎がリンゴとみかんを食べた。二郎は何も食べなかったので腹が減った",
            [["食べた。", "みかんを", "食べた。", ""], ["食べなかったので", "食べなかったので", "減った", "減った", ""]],
        )
    ],
)
def test_knp_parent_getter(nlp: Language, text: str, parents: List[List[str]]):
    doc: Doc = nlp(text)
    for sent, pl in zip(doc.sents, parents):
        spans = sent._.get(KNP_USER_KEYS.tag.spans)
        ps = [span._.get(KNP_USER_KEYS.tag.parent) for span in spans]
        assert [s.text if s else "" for s in ps] == [p for p in pl]


@pytest.mark.parametrize(
    "text,children_list",
    [
        (
            "太郎がリンゴとみかんを食べた。二郎は何も食べなかったので腹が減った",
            [
                [[], [], ["リンゴと"], ["太郎が", "みかんを"]],
                [[], [], ["二郎は", "何も"], [], ["食べなかったので", "腹が"]],
            ],
        )
    ],
)
def test_knp_children_getter(
    nlp: Language, text: str, children_list: List[List[List[str]]]
):
    doc: Doc = nlp(text)
    for sent, children_texts in zip(doc.sents, children_list):
        spans = sent._.get(KNP_USER_KEYS.tag.spans)
        children = [span._.get(KNP_USER_KEYS.tag.children) for span in spans]
        children = [[cc.text for cc in c] for c in children]
        assert children == children_texts
