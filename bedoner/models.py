import mojimoji
import json
from bedoner.entity_extractors.bert_modeling import BertModel
from bedoner.entity_extractors.bert_ner import BertEntityExtractor, create_estimator
from bedoner.lang.mecab import Japanese as Mecab
from bedoner.lang.juman import Japanese as Juman
from bedoner.lang.knp import Japanese as KNP
from pathlib import Path
from pathlib import Path
from spacy.strings import StringStore
from spacy.vocab import Vocab
from bedoner.wordpiecer import BertWordPiecer


__dir__ = Path(__file__).parent


def bert_wordpiecer():
    with (__dir__ / "../data/Japanese_L-12_H-768_A-12_E-30_BPE/vocab.txt").open() as f:
        vs = []
        for line in f:
            vs.append(line[:-1])
    s = StringStore(vs)
    v = Vocab(strings=s)
    nlp = Juman(v, meta={"tokenizer": {"preprocessor": mojimoji.han_to_zen}})
    w = BertWordPiecer(
        v,
        vocab_file=str(__dir__ / "../data/Japanese_L-12_H-768_A-12_E-30_BPE/vocab.txt"),
    )
    w.model = w.Model(w.cfg["vocab_file"])
    nlp.add_pipe(w)
    return nlp


def bert_ner():
    nlp = bert_wordpiecer()

    bert_dir = __dir__ / "../data/Japanese_L-12_H-768_A-12_E-30_BPE"
    model_dir = __dir__ / "../data/bert_result_ene_0/"
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

    ee = BertEntityExtractor.from_nlp(nlp, label2id=label2id, **bert_cfg)
    ee.model = create_estimator(**bert_cfg)
    ee.set_values()
    ee.create_predictor()
    nlp.add_pipe(ee)
    return nlp
