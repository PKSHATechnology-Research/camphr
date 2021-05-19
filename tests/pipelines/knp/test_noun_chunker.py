from typing import List

import pytest
from spacy.language import Language

from camphr_pipelines.models import load
from camphr_test.utils import check_knp

pytestmark = pytest.mark.skipif(not check_knp(), reason="knp is not always necessary")


@pytest.fixture
def nlp():
    _nlp = load("knp")
    _nlp.add_pipe(_nlp.create_pipe("knp_parallel_noun_chunker"))
    return _nlp


@pytest.mark.parametrize(
    "name,text,chunks",
    [
        (0, "望月教授は、数理解析研究所には優れた研究者たちがたくさん在籍する、と述べた。", ["望月教授", "数理解析研究所", "優れた研究者"]),
        (1, "金の斧と銀の斧を持つ", ["金の斧", "銀の斧"]),
        (
            2,
            "新型コロナウイルス感染症の急速な拡大を踏まえ、安倍晋三首相は、新型インフルエンザ等対策特別措置法に基づく緊急事態宣言を発令する方針を固めた。",
            ["新型コロナウイルス感染症の急速な拡大", "安倍晋三首相", "新型インフルエンザ等対策特別措置法に基づく緊急事態宣言を発令する方針"],
        ),
        (3, "金の斧と銀の斧と銅の斧が欲しい", ["金の斧", "銀の斧", "銅の斧"]),
        (4, "りんごとみかんのケーキを食べる", ["りんごとみかんのケーキ"]),
        (5, "りんごとみかんを食べる", ["りんご", "みかん"]),
        (6, "りんごとみかんの重さとぶどうの重さは同じだ", ["りんごとみかんの重さ", "ぶどうの重さ"]),
        (7, "飢えと寒さ", ["飢え", "寒さ"]),
    ],
)
def test_noun_chunker(nlp: Language, text: str, chunks: List[str], name):
    doc = nlp(text)
    assert [s.text for s in doc.noun_chunks] == chunks


@pytest.mark.parametrize(
    "name,text,chunks",
    [
        (0, "望月教授は、数理解析研究所には優れた研究者たちがたくさん在籍する、と述べた。", []),
        (0, "金の斧と銀の斧を持つ", [["金の斧", "銀の斧"]]),
        (0, "金の斧と銀の斧と銅の斧で攻撃する", [["金の斧", "銀の斧", "銅の斧"]]),
        (0, "金の斧と銀の斧と銅の斧を持って，飢えと寒さで死んでいた", [["金の斧", "銀の斧", "銅の斧"], ["飢え", "寒さ"]]),
        (0, "金の斧と銀の斧の職人と銅の斧の職人", [["金の斧と銀の斧の職人", "銅の斧の職人"]]),
    ],
)
def test_para_noun_chunker(nlp: Language, text: str, chunks: List[str], name):
    doc = nlp(text)
    assert [
        [span.text for span in spans] for spans in doc._.get("knp_parallel_noun_chunks")
    ] == chunks
