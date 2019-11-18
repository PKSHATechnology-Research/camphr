import numpy as np
import pytest
import torch
from spacy.language import Language
from spacy.tokens import Doc
from spacy_transformers.util import ATTRS

from bedoner.models import trf_model


@pytest.fixture(params=["mecab", "juman", "sentencepiece"])
def nlp(request, trf_dir):
    lang = request.param
    return trf_model(lang, trf_dir)


TESTCASES = ["今日はいい天気です", "今日は　いい天気です"]


@pytest.mark.parametrize("text", TESTCASES)
def test_forward(nlp, text):
    doc = nlp(text)
    assert doc._.trf_last_hidden_state is not None


@pytest.mark.parametrize("text", TESTCASES)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda test")
def test_forward_cuda(nlp, text):
    assert nlp.to(torch.device("cuda"))
    doc = nlp(text)
    assert doc._.trf_last_hidden_state is not None


@pytest.mark.parametrize("text", TESTCASES)
def test_token_vector(nlp: Language, text: str):
    doc: Doc = nlp(text)
    tensor: torch.Tensor = doc._.get(ATTRS.last_hidden_state).get()
    for token, a in zip(doc, doc._.get(ATTRS.alignment)):
        assert np.allclose(token.vector, tensor[a].sum(0))


@pytest.mark.parametrize("text", TESTCASES)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda test")
def test_token_vector_with_cuda(nlp: Language, text: str):
    assert nlp.to(torch.device("cuda"))
    doc: Doc = nlp(text)
    tensor: torch.Tensor = doc._.get(ATTRS.last_hidden_state).get()
    for token, a in zip(doc, doc._.get(ATTRS.alignment)):
        assert np.allclose(token.vector, tensor[a].sum(0).cpu())


@pytest.mark.parametrize("text", TESTCASES)
def test_span_vector(nlp: Language, text: str):
    doc: Doc = nlp(text)
    assert np.allclose(doc.vector, doc[:].vector)


@pytest.mark.parametrize(
    "text1,text2", [("今日はいい天気だった", "明日は晴れるかな"), ("今日はいい天気だった", "私は自然言語処理マスター")]
)
def test_doc_similarlity(nlp, text1, text2):
    doc1 = nlp(text1)
    doc2 = nlp(text2)
    assert doc1.similarity(doc2)
    assert np.isclose(doc1.similarity(doc2), doc2.similarity(doc1))
