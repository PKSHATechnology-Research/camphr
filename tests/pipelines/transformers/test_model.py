import numpy as np
import omegaconf
import pytest
import torch
from spacy.language import Language
from spacy.tokens import Doc

from camphr.lang.torch import TorchLanguage
from camphr.models import NLPConfig, create_model, load
from camphr.pipelines.transformers.model import TRANSFORMERS_MODEL, TrfModel
from camphr.pipelines.transformers.utils import ATTRS, get_last_hidden_state_from_docs
from camphr.torch_utils import add_loss_to_docs
from tests.utils import TRF_TESTMODEL_PATH, check_serialization

TESTCASES = [
    "今日はいい天気です",
    "今日は　いい天気です",
    "1月16日(木)18時36分頃、沖縄県で最大震度4を観測する地震がありました。",
    "",
    "\n",
]


@pytest.fixture(params=[True, False])
def nlp(nlp_trf_model, request, trf_name_or_path):
    return nlp_trf_model


@pytest.mark.parametrize("text", TESTCASES)
def test_forward(nlp, text):
    doc = nlp(text)
    assert doc._.transformers_last_hidden_state is not None


def test_pipe(nlp):
    list(nlp.pipe(TESTCASES))


@pytest.mark.skip(
    reason="This test fails due to spacy's bug, and will success after the PR is merged: https://github.com/explosion/spaCy/pull/4925"
)
def test_evaluate(nlp: Language):
    docs_golds = [(text, {}) for text in TESTCASES]
    nlp.evaluate(docs_golds, batch_size=1)


@pytest.mark.slow
def test_forward_for_long_input(nlp, lang, trf_name_or_path):
    if lang != "ja_mecab" or trf_name_or_path not in TRF_TESTMODEL_PATH:
        pytest.skip()
    text = "foo " * 2000
    doc = nlp(text)
    assert doc._.transformers_last_hidden_state is not None


@pytest.mark.parametrize("text", TESTCASES)
def test_token_vector(nlp: Language, text: str):
    doc: Doc = nlp(text)
    tensor: torch.Tensor = doc._.get(ATTRS.last_hidden_state).get().cpu()
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


def test_update(nlp: TorchLanguage):
    texts = ["This is a test sentence to check\u3000model.update!"]
    labels = [{}]
    pipe: TrfModel = nlp.get_pipe(TRANSFORMERS_MODEL)
    optimizer = nlp.resume_training()
    eps = 1e-5

    def sum_param(params):
        return sum(p.sum().item() for p in params)

    def train():
        docs, golds = nlp._format_docs_and_golds(texts, labels)
        before = sum_param(pipe.optim_parameters())
        nlp._update_pipes(docs, golds)
        h = get_last_hidden_state_from_docs(docs)
        loss = h.sum() + torch.tensor(0.0, requires_grad=True)
        add_loss_to_docs(docs, loss)
        nlp._update_params(docs, optimizer)
        return abs(before - sum_param(pipe.optim_parameters()))

    assert train() > eps

    # freeze model
    pipe.cfg["freeze"] = True
    assert train() < eps
    # restore freeze state
    pipe.cfg["freeze"] = False


def test_freeze_model(trf_name_or_path, trf_model_config: NLPConfig):
    config = omegaconf.OmegaConf.to_container(trf_model_config)
    config["pipeline"][TRANSFORMERS_MODEL]["freeze"] = True
    nlp = create_model(config)
    pipe = nlp.pipeline[-1][1]
    assert pipe.cfg["freeze"]


def test_check_serialization(nlp):
    check_serialization(nlp)


@pytest.mark.slow
@pytest.mark.parametrize("name", ["bert-base-cased", "xlm-roberta-base"])
def test_load_transformers(name):
    cfg = f"""
    lang:
        name: en
    pipeline:
        transformers_model:
            trf_name_or_path: {name}
    """
    load(cfg)
