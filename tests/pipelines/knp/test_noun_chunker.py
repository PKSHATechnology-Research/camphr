from typing import List

import pytest
from spacy.language import Language

from camphr.models import load

from ...utils import check_knp

pytestmark = pytest.mark.skipif(not check_knp(), reason="knp is not always necessary")


@pytest.fixture
def nlp():
    return load("knp")


@pytest.mark.parametrize(
    "text,chunks",
    [
        ("望月教授は、数理解析研究所には優れた研究者たちがたくさん在籍する、と述べた。", ["望月教授", "数理解析研究所", "優れた研究者"]),
        ("金の斧と銀の斧を持つ", ["金の斧", "銀の斧"]),
        (
            "新型コロナウイルス感染症の急速な拡大を踏まえ、安倍晋三首相は、新型インフルエンザ等対策特別措置法に基づく緊急事態宣言を発令する方針を固めた。",
            ["新型コロナウイルス感染症の急速な拡大", "安倍晋三首相", "新型インフルエンザ等対策特別措置法に基づく緊急事態宣言を発令する方針"],
        ),
    ],
)
def test_noun_chunker(nlp: Language, text: str, chunks: List[str]):
    doc = nlp(text)
    assert [s.text for s in doc.noun_chunks] == chunks
