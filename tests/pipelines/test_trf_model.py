import numpy as np
import pytest
import torch
from camphr.models import trf_model
from camphr.pipelines.trf_model import TransformersModel
from camphr.pipelines.trf_utils import ATTRS
from spacy.language import Language
from spacy.tokens import Doc
from tests.utils import TRF_TESTMODEL_PATH, check_serialization
from transformers import AdamW


@pytest.fixture(scope="session")
def nlp(torch_lang, trf_name_or_path, device):
    _nlp = trf_model(torch_lang, str(trf_name_or_path))
    _nlp.to(device)
    return _nlp


TESTCASES = ["今日はいい天気です", "今日は　いい天気です"]


@pytest.mark.parametrize("text", TESTCASES)
def test_forward(nlp, text):
    doc = nlp(text)
    assert doc._.transformers_last_hidden_state is not None


def test_forward_for_long_input(nlp, torch_lang, trf_name_or_path):
    if torch_lang != "ja_mecab_torch" or trf_name_or_path not in TRF_TESTMODEL_PATH:
        pytest.skip()
    text = "foo " * 2000
    doc = nlp(text)
    assert doc._.transformers_last_hidden_state is not None


@pytest.mark.parametrize("text", TESTCASES)
def test_token_vector(nlp: Language, text: str):
    doc: Doc = nlp(text)
    tensor: torch.Tensor = doc._.get(ATTRS.last_hidden_state).get()
    for token, a in zip(doc, doc._.get(ATTRS.align)):
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


def test_freeze(nlp: Language):
    pipe: TransformersModel = nlp.pipeline[-1][1]
    assert len(list(pipe.optim_parameters())) > 0
    pipe.cfg["freeze"] = True
    assert len(list(pipe.optim_parameters())) == 0
    pipe.cfg["freeze"] = False


def test_optim(nlp: Language):
    optim = nlp.resume_training()
    assert isinstance(optim, AdamW)


def test_freeze_model(trf_testmodel_path):
    nlp = trf_model("ja_mecab_torch", trf_testmodel_path, freeze=True)
    pipe = nlp.pipeline[-1][1]
    assert pipe.cfg["freeze"]


def test_check_serialization(nlp):
    check_serialization(nlp)
