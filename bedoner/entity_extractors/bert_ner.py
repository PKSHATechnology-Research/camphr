from __future__ import annotations
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
from bedore_nlp_modules.bert_modules import modeling
from bedoner.entity_extractors.bert_modeling import BertConfig
from spacy_pytorch_transformers._tokenizers import SerializableBertTokenizer
import tensorflow as tf
import os
from spacy.language import Language
from spacy.tokens import Doc
from spacy.vocab import Vocab


def create_estimator(
    bert_dir: str,
    model_dir: str,
    num_labels: int,
    init_checkpoint: str,
    use_one_hot_embeddings,
    max_seq_length: int,
    batch_size: int = 10,
    id2label=None,
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


class BertEntityExtractor(Pipe):
    @classmethod
    def from_nlp(cls, nlp: Language, **cfg) -> BertEntityExtractor:
        """Factory to add to Language.factories via entry point."""
        return cls(nlp.vocab, **cfg)

    def __init__(
        self,
        vocab: Vocab,
        model=True,
        tokenizer: Optional[SerializableBertTokenizer] = None,
        **cfg,
    ):
        """Initialize the component.

        **cfg: Optional config parameters.
        """
        self.vocab = vocab
        self.model = create_estimator(**cfg)
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.id2label = cfg["id2label"]
        self.max_length = cfg["max_seq_length"]

    def zero_pad(self, a: List[int]):
        if len(a) >= self.max_length:
            return a[: self.max_length]
        return a + [0] * (self.max_length - len(a))

    def __call__(self, doc: Doc) -> Doc:
        """TODO: hash index base"""
        input_fn = self.create_input_fn([doc])
        preds = list(self.model.predict(input_fn))
        return self.set_annotations([doc], preds)[0]

    def set_annotations(
        self, docs: List[Doc], preds: Sequence[Sequence[int]]
    ) -> List[Doc]:
        assert len(docs) == len(preds)
        # TODO: res = [self.tokenizer.convert_ids_to_tokens(r) for r in _res if r != 0]
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

    def create_input_fn(self, docs: List[Doc]) -> Callable:
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
        O, X = (
            self.tokenizer.convert_tokens_to_ids("O"),
            self.tokenizer.convert_tokens_to_ids("X"),
        )
        for doc in docs:
            label_id = [self.tokenizer.cls_token_id]
            for a in doc._.pytt_alignment:
                label_id + [O] + [X] * (len(a) - 1)
            label_ids.append(self.zero_pad(label_id))
        features["label_ids"] = np.array(label_ids)
        return tf.estimator.inputs.numpy_input_fn(features, shuffle=False)


def model_fn_builder(
    bert_config, num_labels, init_checkpoint, use_one_hot_embeddings, max_seq_length
):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

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
        )

        tvars = tf.trainable_variables()
        assignment_map = None
        if init_checkpoint:
            assignment_map, _ = modeling.get_assignment_map_from_checkpoint(
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
    model = modeling.BertModel(
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
