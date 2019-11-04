"""
The base UDify model for training and prediction
"""

import logging
from typing import Any, Dict, List, Optional

import torch
from allennlp.common.checks import ConfigurationError, check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask
from overrides import overrides
from transformers import BertTokenizer

from ..modules.scalar_mix import ScalarMixWithDropout

logger = logging.getLogger(__name__)


class OUTPUTS:
    arc_loss = "arc_loss"
    tag_loss = "tag_loss"
    loss = "loss"
    words = "words"
    ids = "ids"
    multiword_ids = "multiword_ids"
    multiword_forms = "multiword_forms"
    predicted_dependencies = "predicted_dependencies"
    predicted_heads = "predicted_heads"
    feats = "feats"
    lemmas = "lemmas"
    upos = "upos"


@Model.register("udify_model")
class UdifyModel(Model):
    """
    The UDify model base class. Applies a sequence of shared encoders before decoding in a multi-task configuration.
    Uses TagDecoder and DependencyDecoder to decode each UD task.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        tasks: List[str],
        text_field_embedder: TextFieldEmbedder,
        encoder: Seq2SeqEncoder,
        decoders: Dict[str, Model],
        pretrained_model: str,
        post_encoder_embedder: TextFieldEmbedder = None,
        dropout: float = 0.0,
        word_dropout: float = 0.0,
        mix_embedding: int = None,
        layer_dropout: int = 0.0,
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
    ) -> None:
        super(UdifyModel, self).__init__(vocab, regularizer)

        self.tasks = sorted(tasks)
        self.vocab = vocab
        self.bert_vocab = BertTokenizer.from_pretrained(pretrained_model).vocab
        self.text_field_embedder = text_field_embedder
        self.post_encoder_embedder = post_encoder_embedder
        self.shared_encoder = encoder
        self.word_dropout = word_dropout
        self.dropout = torch.nn.Dropout(p=dropout)
        self.decoders = torch.nn.ModuleDict(decoders)

        if mix_embedding:
            self.scalar_mix = torch.nn.ModuleDict(
                {
                    task: ScalarMixWithDropout(
                        mix_embedding, do_layer_norm=False, dropout=layer_dropout
                    )
                    for task in self.decoders
                }
            )
        else:
            self.scalar_mix = None

        self.metrics = {}

        for task in self.tasks:
            if task not in self.decoders:
                raise ConfigurationError(
                    f"Task {task} has no corresponding decoder. Make sure their names match."
                )

        check_dimensions_match(
            text_field_embedder.get_output_dim(),
            encoder.get_input_dim(),
            "text field embedding dim",
            "encoder input dim",
        )

        initializer(self)
        self._count_params()

    @overrides
    def forward(
        self,
        tokens: Dict[str, torch.LongTensor],
        metadata: List[Dict[str, Any]] = None,
        **kwargs: Dict[str, torch.LongTensor],
    ) -> Dict[str, torch.Tensor]:
        if "track_epoch" in kwargs:
            kwargs.pop("track_epoch")

        gold_tags = kwargs

        if "tokens" in self.tasks:
            # Model is predicting tokens, so add them to the gold tags
            gold_tags["tokens"] = tokens["tokens"]

        mask = get_text_field_mask(tokens)
        self._apply_token_dropout(tokens)

        embedded_text_input = self.text_field_embedder(tokens)

        if self.post_encoder_embedder:
            post_embeddings = self.post_encoder_embedder(tokens)

        encoded_text = self.shared_encoder(embedded_text_input, mask)

        logits = {}
        class_probabilities = {}
        output_dict: Dict[str, Any] = {
            "logits": logits,
            "class_probabilities": class_probabilities,
        }
        loss = 0

        # Run through each of the tasks on the shared encoder and save predictions
        for task in self.tasks:
            if self.scalar_mix:
                decoder_input = self.scalar_mix[task](encoded_text, mask)
            else:
                decoder_input = encoded_text

            if self.post_encoder_embedder:
                decoder_input = decoder_input + post_embeddings

            if task == "deps":
                tag_logits = logits["upos"] if "upos" in logits else None
                pred_output = self.decoders[task](
                    decoder_input,
                    mask,
                    tag_logits,
                    gold_tags.get("head_tags", None),
                    gold_tags.get("head_indices", None),
                    metadata,
                )
                for key in ["heads", "head_tags", "arc_loss", "tag_loss", "mask"]:
                    output_dict[key] = pred_output[key]
            else:
                pred_output = self.decoders[task](
                    decoder_input, mask, gold_tags, metadata
                )

                logits[task] = pred_output["logits"]
                class_probabilities[task] = pred_output["class_probabilities"]

            if task in gold_tags or task == "deps" and "head_tags" in gold_tags:
                # Keep track of the loss if we have the gold tags available
                loss += pred_output["loss"]

        if gold_tags:
            output_dict[OUTPUTS.loss] = loss

        if metadata is not None:
            output_dict[OUTPUTS.words] = [x[OUTPUTS.words] for x in metadata]
            output_dict[OUTPUTS.ids] = [
                x[OUTPUTS.ids] for x in metadata if OUTPUTS.ids in x
            ]
            output_dict[OUTPUTS.multiword_ids] = [
                x[OUTPUTS.multiword_ids] for x in metadata if OUTPUTS.multiword_ids in x
            ]
            output_dict[OUTPUTS.multiword_forms] = [
                x[OUTPUTS.multiword_forms]
                for x in metadata
                if OUTPUTS.multiword_forms in x
            ]

        return output_dict

    def _apply_token_dropout(self, tokens):
        # Word dropout
        if "tokens" in tokens:
            oov_token = self.vocab.get_token_index(self.vocab._oov_token)
            ignore_tokens = [self.vocab.get_token_index(self.vocab._padding_token)]
            tokens["tokens"] = self.token_dropout(
                tokens["tokens"],
                oov_token=oov_token,
                padding_tokens=ignore_tokens,
                p=self.word_dropout,
                training=self.training,
            )

        # BERT token dropout
        if "bert" in tokens:
            oov_token = self.bert_vocab["[MASK]"]
            ignore_tokens = [
                self.bert_vocab["[PAD]"],
                self.bert_vocab["[CLS]"],
                self.bert_vocab["[SEP]"],
            ]
            tokens["bert"] = self.token_dropout(
                tokens["bert"],
                oov_token=oov_token,
                padding_tokens=ignore_tokens,
                p=self.word_dropout,
                training=self.training,
            )

    @staticmethod
    def token_dropout(
        tokens: torch.LongTensor,
        oov_token: int,
        padding_tokens: List[int],
        p: float = 0.2,
        training: bool = True,
    ) -> torch.LongTensor:
        """
        During training, randomly replaces some of the non-padding tokens to a mask token with probability ``p``

        :param tokens: The current batch of padded sentences with word ids
        :param oov_token: The mask token
        :param padding_tokens: The tokens for padding the input batch
        :param p: The probability a word gets mapped to the unknown token
        :param training: Applies the dropout if set to ``True``
        :return: A copy of the input batch with token dropout applied
        """
        if training and p > 0:
            # Ensure that the tensors run on the same device
            device = tokens.device

            # This creates a mask that only considers unpadded tokens for mapping to oov
            padding_mask = torch.ones(tokens.size(), dtype=torch.bool).to(device)
            for pad in padding_tokens:
                padding_mask &= tokens != pad

            # Create a uniformly random mask selecting either the original words or OOV tokens
            dropout_mask = (torch.empty(tokens.size()).uniform_() < p).to(device)
            oov_mask = dropout_mask & padding_mask

            oov_fill = (
                torch.empty(tokens.size(), dtype=torch.long).fill_(oov_token).to(device)
            )

            result = torch.where(oov_mask, oov_fill, tokens)

            return result
        else:
            return tokens

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        for task in self.tasks:
            self.decoders[task].decode(output_dict)

        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {
            name: task_metric
            for task in self.tasks
            for name, task_metric in self.decoders[task].get_metrics(reset).items()
        }

        # The "sum" metric summing all tracked metrics keeps a good measure of patience for early stopping and saving
        metrics_to_track = {"upos", "xpos", "feats", "lemmas", "LAS", "UAS"}
        metrics[".run/.sum"] = sum(
            metric
            for name, metric in metrics.items()
            if not name.startswith("_")
            and set(name.split("/")).intersection(metrics_to_track)
        )

        return metrics

    def _count_params(self):
        self.total_params = sum(p.numel() for p in self.parameters())
        self.total_train_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )

        logger.info(f"Total number of parameters: {self.total_params}")
        logger.info(f"Total number of trainable parameters: {self.total_train_params}")
