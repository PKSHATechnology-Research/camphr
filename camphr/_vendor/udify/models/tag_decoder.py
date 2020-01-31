"""
Decodes sequences of tags, e.g., POS tags, given a list of contextualized word embeddings
"""

from typing import Any, Dict, List, Optional

import numpy
import torch
import torch.nn.functional as F
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Seq2SeqEncoder, TimeDistributed
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import sequence_cross_entropy_with_logits
from allennlp.training.metrics import CategoricalAccuracy
from overrides import overrides
from torch.nn.modules.adaptive import AdaptiveLogSoftmaxWithLoss
from torch.nn.modules.linear import Linear

from ..dataset_readers.lemma_edit import apply_lemma_rule


def sequence_cross_entropy(
    log_probs: torch.FloatTensor,
    targets: torch.LongTensor,
    weights: torch.FloatTensor,
    average: str = "batch",
    label_smoothing: float = None,
) -> torch.FloatTensor:
    if average not in {None, "token", "batch"}:
        raise ValueError(
            "Got average f{average}, expected one of " "None, 'token', or 'batch'"
        )
    # shape : (batch * sequence_length, num_classes)
    log_probs_flat = log_probs.view(-1, log_probs.size(2))
    # shape : (batch * max_len, 1)
    targets_flat = targets.view(-1, 1).long()

    if label_smoothing is not None and label_smoothing > 0.0:
        num_classes = log_probs.size(-1)
        smoothing_value = label_smoothing / num_classes
        # Fill all the correct indices with 1 - smoothing value.
        one_hot_targets = torch.zeros_like(log_probs_flat).scatter_(
            -1, targets_flat, 1.0 - label_smoothing
        )
        smoothed_targets = one_hot_targets + smoothing_value
        negative_log_likelihood_flat = -log_probs_flat * smoothed_targets
        negative_log_likelihood_flat = negative_log_likelihood_flat.sum(
            -1, keepdim=True
        )
    else:
        # Contribution to the negative log likelihood only comes from the exact indices
        # of the targets, as the target distributions are one-hot. Here we use torch.gather
        # to extract the indices of the num_classes dimension which contribute to the loss.
        # shape : (batch * sequence_length, 1)
        negative_log_likelihood_flat = -torch.gather(
            log_probs_flat, dim=1, index=targets_flat
        )
    # shape : (batch, sequence_length)
    negative_log_likelihood = negative_log_likelihood_flat.view(*targets.size())
    # shape : (batch, sequence_length)
    negative_log_likelihood = negative_log_likelihood * weights.float()

    if average == "batch":
        # shape : (batch_size,)
        per_batch_loss = negative_log_likelihood.sum(1) / (
            weights.sum(1).float() + 1e-13
        )
        num_non_empty_sequences = (weights.sum(1) > 0).float().sum() + 1e-13
        return per_batch_loss.sum() / num_non_empty_sequences
    elif average == "token":
        return negative_log_likelihood.sum() / (weights.sum().float() + 1e-13)
    else:
        # shape : (batch_size,)
        per_batch_loss = negative_log_likelihood.sum(1) / (
            weights.sum(1).float() + 1e-13
        )
        return per_batch_loss


@Model.register("udify_tag_decoder")
class TagDecoder(Model):
    """
    A basic sequence tagger that decodes from inputs of word embeddings
    """

    def __init__(
        self,
        vocab: Vocabulary,
        task: str,
        encoder: Seq2SeqEncoder,
        label_smoothing: float = 0.0,
        dropout: float = 0.0,
        adaptive: bool = False,
        features: List[str] = None,
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
    ) -> None:
        super(TagDecoder, self).__init__(vocab, regularizer)

        self.task = task
        self.encoder = encoder
        self.output_dim = encoder.get_output_dim()
        self.label_smoothing = label_smoothing
        self.num_classes = self.vocab.get_vocab_size(task)
        self.adaptive = adaptive
        self.features = features if features else []

        self.metrics = {
            "acc": CategoricalAccuracy(),
            # "acc3": CategoricalAccuracy(top_k=3)
        }

        if self.adaptive:
            # TODO
            adaptive_cutoffs = [
                round(self.num_classes / 15),
                3 * round(self.num_classes / 15),
            ]
            self.task_output = AdaptiveLogSoftmaxWithLoss(
                self.output_dim,
                self.num_classes,
                cutoffs=adaptive_cutoffs,
                div_value=4.0,
            )
        else:
            self.task_output = TimeDistributed(
                Linear(self.output_dim, self.num_classes)
            )

        self.feature_outputs = torch.nn.ModuleDict()
        self.features_metrics = {}
        for feature in self.features:
            self.feature_outputs[feature] = TimeDistributed(
                Linear(self.output_dim, vocab.get_vocab_size(feature))
            )
            self.features_metrics[feature] = {"acc": CategoricalAccuracy()}

        initializer(self)

    @overrides
    def forward(
        self,
        encoded_text: torch.FloatTensor,
        mask: torch.LongTensor,
        gold_tags: Dict[str, torch.LongTensor],
        metadata: List[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        hidden = encoded_text
        hidden = self.encoder(hidden, mask)

        batch_size, sequence_length, _ = hidden.size()
        output_dim = [batch_size, sequence_length, self.num_classes]

        loss_fn = self._adaptive_loss if self.adaptive else self._loss

        output_dict = loss_fn(hidden, mask, gold_tags.get(self.task, None), output_dim)
        self._features_loss(hidden, mask, gold_tags, output_dict)

        return output_dict

    def _adaptive_loss(self, hidden, mask, gold_tags, output_dim):
        logits = hidden
        reshaped_log_probs = logits.view(-1, logits.size(2))

        class_probabilities = self.task_output.log_prob(reshaped_log_probs).view(
            output_dim
        )

        output_dict = {"logits": logits, "class_probabilities": class_probabilities}

        if gold_tags is not None:
            output_dict["loss"] = sequence_cross_entropy(
                class_probabilities,
                gold_tags,
                mask,
                label_smoothing=self.label_smoothing,
            )
            for metric in self.metrics.values():
                metric(class_probabilities, gold_tags, mask.float())

        return output_dict

    def _loss(self, hidden, mask, gold_tags, output_dim):
        logits = self.task_output(hidden)
        reshaped_log_probs = logits.view(-1, self.num_classes)
        class_probabilities = F.softmax(reshaped_log_probs, dim=-1).view(output_dim)

        output_dict = {"logits": logits, "class_probabilities": class_probabilities}

        if gold_tags is not None:
            output_dict["loss"] = sequence_cross_entropy_with_logits(
                logits, gold_tags, mask, label_smoothing=self.label_smoothing
            )
            for metric in self.metrics.values():
                metric(logits, gold_tags, mask.float())

        return output_dict

    def _features_loss(self, hidden, mask, gold_tags, output_dict):
        if gold_tags is None:
            return

        for feature in self.features:
            logits = self.feature_outputs[feature](hidden)
            loss = sequence_cross_entropy_with_logits(
                logits, gold_tags[feature], mask, label_smoothing=self.label_smoothing
            )
            loss /= len(self.features)
            output_dict["loss"] += loss

            for metric in self.features_metrics[feature].values():
                metric(logits, gold_tags[feature], mask.float())

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        all_words = output_dict["words"]

        all_predictions = (
            output_dict["class_probabilities"][self.task].cpu().data.numpy()
        )
        if all_predictions.ndim == 3:
            predictions_list = [
                all_predictions[i] for i in range(all_predictions.shape[0])
            ]
        else:
            predictions_list = [all_predictions]
        all_tags = []
        for predictions, words in zip(predictions_list, all_words):
            argmax_indices = numpy.argmax(predictions, axis=-1)
            tags = [
                self.vocab.get_token_from_index(x, namespace=self.task)
                for x in argmax_indices
            ]

            # TODO: specific task
            if self.task == "lemmas":

                def decode_lemma(word, rule):
                    if rule == "_":
                        return "_"
                    if rule == "@@UNKNOWN@@":
                        return word
                    return apply_lemma_rule(word, rule)

                tags = [decode_lemma(word, rule) for word, rule in zip(words, tags)]

            all_tags.append(tags)
        output_dict[self.task] = all_tags

        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        main_metrics = {
            f".run/{self.task}/{metric_name}": metric.get_metric(reset)
            for metric_name, metric in self.metrics.items()
        }

        features_metrics = {
            f"_run/{self.task}/{feature}/{metric_name}": metric.get_metric(reset)
            for feature in self.features
            for metric_name, metric in self.features_metrics[feature].items()
        }

        return {**main_metrics, **features_metrics}
