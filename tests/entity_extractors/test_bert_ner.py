from bedoner.entity_extractors.bert_modeling import BertModel
import spacy
import shutil
import pytest
from bedoner.entity_extractors.bert_ner import BertEntityExtractor, create_estimator
from pathlib import Path
import pickle
from spacy.tokens import Doc
import json
from spacy.language import Language
from numpy.testing import assert_array_almost_equal


@pytest.fixture(scope="module")
def nlp(bert_wordpiece_nlp):
    __dir__ = Path(__file__).parent
    bert_dir = __dir__ / "../../data/Japanese_L-12_H-768_A-12_E-30_BPE"
    model_dir = __dir__ / "../../data/bert_result_ene_0/"
    init_checkpoint = str(bert_dir / "bert_model.ckpt")
    with (model_dir / "label2id.json").open("r") as f:
        label2id = json.load(f)

    bert_cfg = dict(
        bert_dir=str(bert_dir),
        model_dir=str(model_dir),
        num_labels=len(label2id) + 1,
        init_checkpoint=init_checkpoint,
        use_one_hot_embeddings=None,
        max_seq_length=128,
        batch_size=10,
    )

    ee = BertEntityExtractor.from_nlp(bert_wordpiece_nlp, label2id=label2id, **bert_cfg)
    ee.model = create_estimator(**bert_cfg)
    ee.set_values()
    ee.create_predictor()
    bert_wordpiece_nlp.add_pipe(ee)
    return bert_wordpiece_nlp


TESTCASE = [
    (
        "EXILEのATSUSHIと中島美嘉が14日ニューヨーク入り",
        [
            ("ＥＸＩＬＥ", "Show_Organization"),
            ("ＡＴＳＵＳＨＩ", "Person"),
            ("中島美嘉", "Person"),
            ("１４日", "Date"),
            ("ニューヨーク", "City"),
        ],
    )
]


@pytest.fixture
def saved_nlp(nlp):
    nlp.to_disk("./foo")
    nlp = spacy.load("./foo")
    shutil.rmtree("./foo")
    return nlp


@pytest.mark.parametrize("text,ents", TESTCASE)
def test_to_disk(saved_nlp: Language, text, ents):
    doc: Doc = saved_nlp(text)
    for pred, ans in zip(doc.ents, ents):
        assert pred.text == ans[0]
        assert pred.label_ == ans[1]

