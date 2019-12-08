import numpy as np
import pytest
import spacy
import torch
from bedoner.models import trf_model
from bedoner.pipelines.trf_model import TransformersModel
from spacy.language import Language
from spacy.tokens import Doc
from spacy_transformers.util import ATTRS
from transformers import AdamW


@pytest.fixture
def nlp(lang, trf_dir, device):
    _nlp = trf_model(lang, trf_dir)
    _nlp.to(device)
    return _nlp


TESTCASES = ["今日はいい天気です", "今日は　いい天気です"]


@pytest.mark.parametrize("text", TESTCASES)
def test_forward(nlp, text):
    doc = nlp(text)
    assert doc._.trf_last_hidden_state is not None


def test_forward_for_long_input(nlp, lang):
    if lang != "mecab":
        pytest.skip()
    text = "foo " * 2000
    doc = nlp(text)
    assert doc._.trf_last_hidden_state is not None


@pytest.mark.parametrize("text", TESTCASES)
def test_token_vector(nlp: Language, text: str):
    doc: Doc = nlp(text)
    tensor: torch.Tensor = doc._.get(ATTRS.last_hidden_state).get()
    for token, a in zip(doc, doc._.get(ATTRS.alignment)):
        assert np.allclose(token.vector, tensor[a].sum(0))


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


def test_freeze(nlp: Language):
    pipe: TransformersModel = nlp.pipeline[-1][1]
    assert len(list(pipe.optim_parameters())) > 0
    pipe.cfg["freeze"] = True
    assert len(list(pipe.optim_parameters())) == 0


def test_optim(nlp: Language):
    optim = nlp.resume_training()
    assert isinstance(optim, AdamW)


@pytest.mark.xfail(reason="after feature/trf-maskedlm merged")
def test_update(nlp: Language, tmpdir):
    optim = nlp.resume_training()
    nlp.update(TESTCASES, [{}] * len(TESTCASES), optim)
    nlp.to_disk(str(tmpdir))
    nlp = spacy.load(str(tmpdir))

    optim = nlp.resume_training()
    nlp.update(TESTCASES, [{}] * len(TESTCASES), optim)
