from __future__ import annotations
from os.path import exists
import tensorflow
from tensorflow.contrib import predictor
import json
import shutil
from copy import copy
from distutils.dir_util import copy_tree
from typing import *
import itertools as it
import pickle
from pathlib import Path

import mojimoji
import numpy as np
from spacy.strings import StringStore
from spacy.tokens import Span
from spacy.vocab import Vocab
from spacy.pipeline import Pipe

from bedoner.lang.juman import Japanese
from bedoner.entity_extractors.bert_modeling import (
    BertConfig,
    get_assignment_map_from_checkpoint,
    BertConfig,
    BertModel,
)
from spacy_pytorch_transformers._tokenizers import SerializableBertTokenizer
import tensorflow as tf
import os
from spacy.language import Language
from spacy.tokens import Doc
from spacy.vocab import Vocab


class BertEntityExtractor(Pipe):
    name = "bert_entity_extractor"
    BERT_DIR = "bert_dir"
    MODEL_DIR = "model_dir"
    CHECKPOINT = "bert_model.ckpt"
    BERT_CFG_JSON = "bert_cfg.json"
    BERT_CFG_FIEZLDS = [
        "num_labels",
        "use_one_hot_embeddings",
        "max_seq_length",
        "batch_size",
    ]

    @classmethod
    def from_nlp(cls, nlp: Language, **cfg) -> BertEntityExtractor:
        """Factory to add to Language.factories via entry point."""
        return cls(nlp.vocab, **cfg)

    def __init__(self, vocab: Vocab, model=True, **cfg):
        """Initialize the component.

        **cfg: Optional config parameters.
        """
        self.cfg = cfg
        self.model: tf.estimator.Estimator = model
        self.vocab = vocab

    def __call__(self, doc: Doc) -> Doc:
        features = self.create_features([doc])
        preds = self.predictor(features)["output"]
        return self.set_annotations([doc], preds)[0]

    def set_annotations(
        self, docs: List[Doc], preds: Sequence[Sequence[int]]
    ) -> List[Doc]:
        assert len(docs) == len(preds)
        labels_list = [[self.id2label[r] for r in line if r != 0] for line in preds]

        for doc, labels in zip(docs, labels_list):
            new_ents = []
            cur = []
            for i, a in enumerate(doc._.pytt_alignment):
                l = labels[a[0]]
                doc[i].ent_type_ = l
                if l.startswith("B-"):
                    if cur:
                        new_ents.append(Span(doc, cur[0], cur[1], cur[2]))
                    cur = [i, i + 1, l.lstrip("B-")]
                elif l.startswith("I-"):
                    cur[1] += 1
            if cur:
                new_ents.append(Span(doc, cur[0], cur[1], cur[2]))
            doc.ents = new_ents
        return docs

    def create_features(self, docs: List[Doc]) -> Dict:
        features = {}
        features["input_ids"] = np.array(
            [self.zero_pad(doc._.pytt_word_pieces) for doc in docs]
        )
        features["input_mask"] = np.array(
            [
                self.zero_pad([1 for _ in range(len(doc._.pytt_word_pieces))])
                for doc in docs
            ]
        )
        segs = []
        for doc in docs:
            seg = []
            for i, s in enumerate(doc._.pytt_segments):
                seg += [i] * len(s._.pytt_word_pieces)
            segs.append(self.zero_pad(seg))
        features["segment_ids"] = np.array(segs)
        label_ids = []
        O, X = self.label2id["O"], self.label2id["X"]
        for doc in docs:
            label_id = [self.label2id["[CLS]"]]  # TODO* avoid hard code
            for a in doc._.pytt_alignment:
                label_id + [O] + [X] * (len(a) - 1)
            label_ids.append(self.zero_pad(label_id))
        features["label_ids"] = np.array(label_ids)
        return features

    def zero_pad(self, a: List[int]):
        if len(a) >= self.max_length:
            return a[: self.max_length]
        return a + [0] * (self.max_length - len(a))

    def to_disk(self, path: Path, exclude=tuple(), **kwargs):
        """Serialize the pipe to disk."""
        path.mkdir(exist_ok=True)
        copy_tree(self.cfg[self.BERT_DIR], str(path / self.BERT_DIR))
        copy_tree(self.cfg[self.MODEL_DIR], str(path / self.MODEL_DIR))
        bert_cfgs = {k: self.cfg[k] for k in self.BERT_CFG_FIEZLDS}
        with (path / self.BERT_CFG_JSON).open("w") as f:
            json.dump(bert_cfgs, f)

        with (path / "label2id.json").open("w") as f:
            json.dump(self.label2id, f)

    def from_disk(self, path, exclude=tuple(), **kwargs):
        self.cfg = self.cfg or {}
        with (path / "label2id.json").open() as f:
            self.cfg["label2id"] = json.load(f)
        with (path / self.BERT_CFG_JSON).open() as f:
            bert_cfg = json.load(f)
        for k, v in bert_cfg.items():
            self.cfg[k] = v
        self.cfg[self.BERT_DIR] = bert_cfg["bert_dir"] = str(path / self.BERT_DIR)
        self.cfg[self.MODEL_DIR] = bert_cfg["model_dir"] = str(path / self.MODEL_DIR)
        self.cfg["init_checkpoint"] = bert_cfg["init_checkpoint"] = str(
            path / self.BERT_DIR / self.CHECKPOINT
        )

        self.model = create_estimator(**bert_cfg)
        self.set_values()
        self.create_predictor()

    def create_predictor(self):
        self.predictor = predictor.from_estimator(
            self.model, create_serving_reciever_fn(self.max_length)
        )

    def set_values(self):
        self.label2id: Dict[str, int] = self.cfg["label2id"]
        self.id2label: Dict[int, str] = {v: k for k, v in self.label2id.items()}
        self.max_length = self.cfg["max_seq_length"]


def create_serving_reciever_fn(seq_length):
    def serving_input_receiver_fn():
        feature_spec = {
            "label_ids": tf.placeholder(
                dtype=tf.int32, shape=[None, seq_length], name="label_ids"
            ),
            "input_ids": tf.placeholder(
                dtype=tf.int32, shape=[None, seq_length], name="input_ids"
            ),
            "input_mask": tf.placeholder(
                dtype=tf.int32, shape=[None, seq_length], name="input_mask"
            ),
            "segment_ids": tf.placeholder(
                dtype=tf.int32, shape=[None, seq_length], name="segment_ids"
            ),
        }
        return tf.estimator.export.ServingInputReceiver(feature_spec, feature_spec)

    return serving_input_receiver_fn


def create_estimator(
    bert_dir: str,
    model_dir: str,
    num_labels: int,
    init_checkpoint: str,
    use_one_hot_embeddings: bool,
    max_seq_length: int,
    batch_size: int,
) -> tf.estimator.Estimator:
    bert_config = BertConfig.from_json_file(os.path.join(bert_dir, "bert_config.json"))
    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=num_labels,
        init_checkpoint=init_checkpoint,
        use_one_hot_embeddings=use_one_hot_embeddings,
        max_seq_length=max_seq_length,
    )

    run_config = tf.estimator.RunConfig(model_dir=model_dir)
    return tf.estimator.Estimator(
        model_fn=model_fn, config=run_config, params={"batch_size": batch_size}
    )


def model_fn_builder(
    bert_config: tf.estimator.RunConfig,
    num_labels: int,
    init_checkpoint: str,
    use_one_hot_embeddings: bool,
    max_seq_length: int,
) -> Callable:
    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]

        is_training = mode == tf.estimator.ModeKeys.TRAIN

        (total_loss, per_example_loss, logits, predicts) = create_model(
            bert_config,
            is_training,
            input_ids,
            input_mask,
            segment_ids,
            label_ids,
            num_labels,
            use_one_hot_embeddings,
            max_seq_length,
        )  # pylint: disable=unused-variable

        tvars = tf.trainable_variables()
        assignment_map = None
        if init_checkpoint:
            assignment_map, _ = get_assignment_map_from_checkpoint(
                tvars, init_checkpoint
            )

        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        output_spec = None
        if mode == tf.estimator.ModeKeys.PREDICT:
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode, predictions=predicts  # {"probabilities": probabilities},
            )
        return output_spec

    return model_fn


def create_model(
    bert_config,
    is_training,
    input_ids,
    input_mask,
    segment_ids,
    labels,
    num_labels,
    use_one_hot_embeddings,
    max_seq_length,
):
    """Creates a token-level classification model."""
    model = BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings,
    )

    output_layer = model.get_sequence_output()

    hidden_size = output_layer.shape[-1].value

    output_weights = tf.get_variable(
        "output_weights",
        [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02),
    )

    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer()
    )

    with tf.variable_scope("loss"):
        if is_training:
            # I.e., 0.1 dropout
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
        output_layer = tf.reshape(output_layer, [-1, hidden_size])

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        logits = tf.reshape(logits, [-1, max_seq_length, num_labels])
        probabilities = tf.nn.softmax(logits, axis=-1)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)

        predict = tf.argmax(probabilities, axis=-1)

        return (loss, per_example_loss, logits, predict)


Language.factories[BertEntityExtractor.name] = BertEntityExtractor

