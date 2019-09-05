import pytest
from bedoner.entity_extractors.bert_ner import BertEntityExtractor
from pathlib import Path
import pickle
from spacy.tokens import Doc


@pytest.fixture
def nlp(bert_wordpiece_nlp):
    __dir__ = Path(__file__).parent
    bert_dir = __dir__ / "../../data/Japanese_L-12_H-768_A-12_E-30_BPE"
    model_dir = __dir__ / "../../data/bert_result_ene_0/"
    init_checkpoint = str(bert_dir / "bert_model.ckpt")
    with (model_dir / "id2label.pkl").open("rb") as f:
        id2label = pickle.load(f)

    ee = BertEntityExtractor.from_nlp(
        bert_wordpiece_nlp,
        tokenizer=bert_wordpiece_nlp.get_pipe("pytt_wordpiecer").model,
        bert_dir=str(bert_dir),
        model_dir=str(model_dir),
        num_labels=len(id2label) + 1,
        init_checkpoint=init_checkpoint,
        use_one_hot_embeddings=None,
        max_seq_length=128,
        id2label=id2label,
    )
    bert_wordpiece_nlp.add_pipe(ee)
    return bert_wordpiece_nlp


@pytest.mark.parametrize(
    "text,ents",
    [
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
    ],
)
def test_bert_ner(nlp, text, ents):
    doc: Doc = nlp(text)
    for pred, ans in zip(doc.ents, ents):
        assert pred.text == ans[0]
        assert pred.label_ == ans[1]
