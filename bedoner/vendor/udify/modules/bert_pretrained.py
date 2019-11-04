"""
An extension of AllenNLP's BERT pretrained helper classes. Supports modification to BERT dropout, and applies
a sliding window approach for long sentences.
"""

import logging
from typing import Callable, Dict, List

import torch
import torch.nn.functional as F
from allennlp.common.util import pad_sequence_to_length
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from allennlp.data.tokenizers.token import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.token_embedders import TokenEmbedder
from allennlp.nn import util
from overrides import overrides

from transformers import BertConfig, BertModel, BertTokenizer

from .scalar_mix import ScalarMixWithDropout

logger = logging.getLogger(__name__)

# TODO(joelgrus): Figure out how to generate token_type_ids out of this token indexer.

# This is the default list of tokens that should not be lowercased.
_NEVER_LOWERCASE = ["[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"]


class WordpieceIndexer(TokenIndexer[int]):
    """
    A token indexer that does the wordpiece-tokenization (e.g. for BERT embeddings).
    If you are using one of the pretrained BERT models, you'll want to use the ``PretrainedBertIndexer``
    subclass rather than this base class.
    Parameters
    ----------
    vocab : ``Dict[str, int]``
        The mapping {wordpiece -> id}.  Note this is not an AllenNLP ``Vocabulary``.
    wordpiece_tokenizer : ``Callable[[str], List[str]]``
        A function that does the actual tokenization.
    namespace : str, optional (default: "wordpiece")
        The namespace in the AllenNLP ``Vocabulary`` into which the wordpieces
        will be loaded.
    use_starting_offsets : bool, optional (default: False)
        By default, the "offsets" created by the token indexer correspond to the
        last wordpiece in each word. If ``use_starting_offsets`` is specified,
        they will instead correspond to the first wordpiece in each word.
    max_pieces : int, optional (default: 512)
        The BERT embedder uses positional embeddings and so has a corresponding
        maximum length for its input ids. Any inputs longer than this will
        either be truncated (default), or be split apart and batched using a
        sliding window.
    do_lowercase : ``bool``, optional (default=``False``)
        Should we lowercase the provided tokens before getting the indices?
        You would need to do this if you are using an -uncased BERT model
        but your DatasetReader is not lowercasing tokens (which might be the
        case if you're also using other embeddings based on cased tokens).
    never_lowercase: ``List[str]``, optional
        Tokens that should never be lowercased. Default is
        ['[UNK]', '[SEP]', '[PAD]', '[CLS]', '[MASK]'].
    start_tokens : ``List[str]``, optional (default=``None``)
        These are prepended to the tokens provided to ``tokens_to_indices``.
    end_tokens : ``List[str]``, optional (default=``None``)
        These are appended to the tokens provided to ``tokens_to_indices``.
    separator_token : ``str``, optional (default=``[SEP]``)
        This token indicates the segments in the sequence.
    truncate_long_sequences : ``bool``, optional (default=``True``)
        By default, long sequences will be truncated to the maximum sequence
        length. Otherwise, they will be split apart and batched using a
        sliding window.
    """

    def __init__(
        self,
        vocab: Dict[str, int],
        wordpiece_tokenizer: Callable[[str], List[str]],
        namespace: str = "wordpiece",
        use_starting_offsets: bool = False,
        max_pieces: int = 512,
        do_lowercase: bool = False,
        never_lowercase: List[str] = None,
        start_tokens: List[str] = None,
        end_tokens: List[str] = None,
        separator_token: str = "[SEP]",
        truncate_long_sequences: bool = True,
        token_min_padding_length: int = 0,
    ) -> None:

        super().__init__(token_min_padding_length)
        self.vocab = vocab

        # The BERT code itself does a two-step tokenization:
        #    sentence -> [words], and then word -> [wordpieces]
        # In AllenNLP, the first step is implemented as the ``BertSimpleWordSplitter``,
        # and this token indexer handles the second.
        self.wordpiece_tokenizer = wordpiece_tokenizer

        self._namespace = namespace
        self._added_to_vocabulary = False
        self.max_pieces = max_pieces
        self.use_starting_offsets = use_starting_offsets
        self._do_lowercase = do_lowercase
        self._truncate_long_sequences = truncate_long_sequences

        if never_lowercase is None:
            # Use the defaults
            self._never_lowercase = set(_NEVER_LOWERCASE)
        else:
            self._never_lowercase = set(never_lowercase)

        # Convert the start_tokens and end_tokens to wordpiece_ids
        self._start_piece_ids = [
            vocab[wordpiece]
            for token in (start_tokens or [])
            for wordpiece in wordpiece_tokenizer(token)
        ]
        self._end_piece_ids = [
            vocab[wordpiece]
            for token in (end_tokens or [])
            for wordpiece in wordpiece_tokenizer(token)
        ]

        # Convert the separator_token to wordpiece_ids
        self._separator_ids = [
            vocab[wordpiece] for wordpiece in wordpiece_tokenizer(separator_token)
        ]

    @overrides
    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]):
        # If we only use pretrained models, we don't need to do anything here.
        pass

    def _add_encoding_to_vocabulary(self, vocabulary: Vocabulary) -> None:
        # pylint: disable=protected-access
        for word, idx in self.vocab.items():
            vocabulary._token_to_index[self._namespace][word] = idx
            vocabulary._index_to_token[self._namespace][idx] = word

    @overrides
    def tokens_to_indices(
        self, tokens: List[Token], vocabulary: Vocabulary, index_name: str
    ) -> Dict[str, List[List[int]]]:
        if not self._added_to_vocabulary:
            self._add_encoding_to_vocabulary(vocabulary)
            self._added_to_vocabulary = True

        # This lowercases tokens if necessary
        text = (
            token.text.lower()
            if self._do_lowercase and token.text not in self._never_lowercase
            else token.text
            for token in tokens
        )

        # Obtain a nested sequence of wordpieces, each represented by a list of wordpiece ids
        token_wordpiece_ids = [
            [self.vocab[wordpiece] for wordpiece in self.wordpiece_tokenizer(token)]
            for token in text
        ]

        # offsets[i] will give us the index into wordpiece_ids
        # for the wordpiece "corresponding to" the i-th input token.
        offsets = []

        # If we're using initial offsets, we want to start at offset = len(text_tokens)
        # so that the first offset is the index of the first wordpiece of tokens[0].
        # Otherwise, we want to start at len(text_tokens) - 1, so that the "previous"
        # offset is the last wordpiece of "tokens[-1]".
        offset = (
            len(self._start_piece_ids)
            if self.use_starting_offsets
            else len(self._start_piece_ids) - 1
        )

        for token in token_wordpiece_ids:
            # For initial offsets, the current value of ``offset`` is the start of
            # the current wordpiece, so add it to ``offsets`` and then increment it.
            if self.use_starting_offsets:
                offsets.append(offset)
                offset += len(token)
            # For final offsets, the current value of ``offset`` is the end of
            # the previous wordpiece, so increment it and then add it to ``offsets``.
            else:
                offset += len(token)
                offsets.append(offset)

        # Flattened list of wordpieces. In the end, the output of the model (e.g., BERT) should
        # have a sequence length equal to the length of this list. However, it will first be split into
        # chunks of length `self.max_pieces` so that they can be fit through the model. After packing
        # and passing through the model, it should be unpacked to represent the wordpieces in this list.
        flat_wordpiece_ids = [
            wordpiece for token in token_wordpiece_ids for wordpiece in token
        ]

        # The code below will (possibly) pack the wordpiece sequence into multiple sub-sequences by using a sliding
        # window `window_length` that overlaps with previous windows according to the `stride`. Suppose we have
        # the following sentence: "I went to the store to buy some milk". Then a sliding window of length 4 and
        # stride of length 2 will split them up into:

        # "[I went to the] [to the store to] [store to buy some] [buy some milk [PAD]]".

        # This is to ensure that the model has context of as much of the sentence as possible to get accurate
        # embeddings. Finally, the sequences will be padded with any start/end piece ids, e.g.,

        # "[CLS] I went to the [SEP] [CLS] to the store to [SEP] ...".

        # The embedder should then be able to split this token sequence by the window length,
        # pass them through the model, and recombine them.

        # Specify the stride to be half of `self.max_pieces`, minus any additional start/end wordpieces
        window_length = (
            self.max_pieces - len(self._start_piece_ids) - len(self._end_piece_ids)
        )
        stride = window_length // 2

        if len(flat_wordpiece_ids) <= window_length:
            # If all the wordpieces fit, then we don't need to do anything special
            wordpiece_windows = [
                self._start_piece_ids + flat_wordpiece_ids + self._end_piece_ids
            ]
        elif self._truncate_long_sequences:
            logger.warning(
                "Too many wordpieces, truncating sequence. If you would like a rolling window, set"
                "`truncate_long_sequences` to False"
                f"{[token.text for token in tokens]}"
            )
            wordpiece_windows = [
                self._start_piece_ids
                + flat_wordpiece_ids[:window_length]
                + self._end_piece_ids
            ]
        else:
            # Create a sliding window of wordpieces of length `max_pieces` that advances by `stride` steps and
            # add start/end wordpieces to each window
            # TODO: this currently does not respect word boundaries, so words may be cut in half between windows
            # However, this would increase complexity, as sequences would need to be padded/unpadded in the middle
            wordpiece_windows = [
                self._start_piece_ids
                + flat_wordpiece_ids[i : i + window_length]
                + self._end_piece_ids
                for i in range(0, len(flat_wordpiece_ids), stride)
            ]

            # Check for overlap in the last window. Throw it away if it is redundant.
            last_window = wordpiece_windows[-1][1:]
            penultimate_window = wordpiece_windows[-2]
            if last_window == penultimate_window[-len(last_window) :]:
                wordpiece_windows = wordpiece_windows[:-1]

        # Flatten the wordpiece windows
        wordpiece_ids = [
            wordpiece for sequence in wordpiece_windows for wordpiece in sequence
        ]

        # Constructing `token_type_ids` by `self._separator`
        token_type_ids = _get_token_type_ids(wordpiece_ids, self._separator_ids)

        # Our mask should correspond to the original tokens,
        # because calling util.get_text_field_mask on the
        # "wordpiece_id" tokens will produce the wrong shape.
        # However, because of the max_pieces constraint, we may
        # have truncated the wordpieces; accordingly, we want the mask
        # to correspond to the remaining tokens after truncation, which
        # is captured by the offsets.
        mask = [1 for _ in offsets]

        return {
            index_name: wordpiece_ids,
            f"{index_name}-offsets": offsets,
            f"{index_name}-type-ids": token_type_ids,
            "mask": mask,
        }

    @overrides
    def get_padding_token(self) -> int:
        return 0

    @overrides
    def get_padding_lengths(
        self, token: int
    ) -> Dict[str, int]:  # pylint: disable=unused-argument
        return {}

    @overrides
    def pad_token_sequence(
        self,
        tokens: Dict[str, List[int]],
        desired_num_tokens: Dict[str, int],
        padding_lengths: Dict[str, int],
    ) -> Dict[str, List[int]]:  # pylint: disable=unused-argument
        return {
            key: pad_sequence_to_length(val, desired_num_tokens[key])
            for key, val in tokens.items()
        }

    @overrides
    def get_keys(self, index_name: str) -> List[str]:
        """
        We need to override this because the indexer generates multiple keys.
        """
        # pylint: disable=no-self-use
        return [index_name, f"{index_name}-offsets", f"{index_name}-type-ids", "mask"]


@TokenIndexer.register("udify-bert-pretrained")
class PretrainedBertIndexer(WordpieceIndexer):
    # pylint: disable=line-too-long
    """
    A ``TokenIndexer`` corresponding to a pretrained BERT model.

    Parameters
    ----------
    pretrained_model: ``str``
        Either the name of the pretrained model to use (e.g. 'bert-base-uncased'),
        or the path to the .txt file with its vocabulary.

        If the name is a key in the list of pretrained models at
        https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/pytorch_pretrained_bert/tokenization.py#L33
        the corresponding path will be used; otherwise it will be interpreted as a path or URL.
    use_starting_offsets: bool, optional (default: False)
        By default, the "offsets" created by the token indexer correspond to the
        last wordpiece in each word. If ``use_starting_offsets`` is specified,
        they will instead correspond to the first wordpiece in each word.
    do_lowercase: ``bool``, optional (default = True)
        Whether to lowercase the tokens before converting to wordpiece ids.
    never_lowercase: ``List[str]``, optional
        Tokens that should never be lowercased. Default is
        ['[UNK]', '[SEP]', '[PAD]', '[CLS]', '[MASK]'].
    max_pieces: int, optional (default: 512)
        The BERT embedder uses positional embeddings and so has a corresponding
        maximum length for its input ids. Any inputs longer than this will
        either be truncated (default), or be split apart and batched using a
        sliding window.
    truncate_long_sequences : ``bool``, optional (default=``True``)
        By default, long sequences will be truncated to the maximum sequence
        length. Otherwise, they will be split apart and batched using a
        sliding window.
    """

    def __init__(
        self,
        pretrained_model: str,
        use_starting_offsets: bool = False,
        do_lowercase: bool = True,
        never_lowercase: List[str] = None,
        max_pieces: int = 512,
        truncate_long_sequences: bool = False,
    ) -> None:
        if pretrained_model.endswith("-cased") and do_lowercase:
            logger.warning(
                "Your BERT model appears to be cased, "
                "but your indexer is lowercasing tokens."
            )
        elif pretrained_model.endswith("-uncased") and not do_lowercase:
            logger.warning(
                "Your BERT model appears to be uncased, "
                "but your indexer is not lowercasing tokens."
            )

        bert_tokenizer = BertTokenizer.from_pretrained(
            pretrained_model, do_lower_case=do_lowercase
        )
        super().__init__(
            vocab=bert_tokenizer.vocab,
            wordpiece_tokenizer=bert_tokenizer.wordpiece_tokenizer.tokenize,
            namespace="bert",
            use_starting_offsets=use_starting_offsets,
            max_pieces=max_pieces,
            do_lowercase=do_lowercase,
            never_lowercase=never_lowercase,
            start_tokens=["[CLS]"],
            end_tokens=["[SEP]"],
            separator_token="[SEP]",
            truncate_long_sequences=truncate_long_sequences,
        )


def _get_token_type_ids(
    wordpiece_ids: List[int], separator_ids: List[int]
) -> List[int]:
    num_wordpieces = len(wordpiece_ids)
    token_type_ids: List[int] = []
    type_id = 0
    cursor = 0
    while cursor < num_wordpieces:
        # check length
        if num_wordpieces - cursor < len(separator_ids):
            token_type_ids.extend(type_id for _ in range(num_wordpieces - cursor))
            cursor += num_wordpieces - cursor
        # check content
        # when it is a separator
        elif all(
            wordpiece_ids[cursor + index] == separator_id
            for index, separator_id in enumerate(separator_ids)
        ):
            token_type_ids.extend(type_id for _ in separator_ids)
            type_id += 1
            cursor += len(separator_ids)
        # when it is not
        else:
            cursor += 1
            token_type_ids.append(type_id)
    return token_type_ids


class BertEmbedder(TokenEmbedder):
    """
    A ``TokenEmbedder`` that produces BERT embeddings for your tokens.
    Should be paired with a ``BertIndexer``, which produces wordpiece ids.
    Most likely you probably want to use ``PretrainedBertEmbedder``
    for one of the named pretrained models, not this base class.
    Parameters
    ----------
    bert_model: ``BertModel``
        The BERT model being wrapped.
    top_layer_only: ``bool``, optional (default = ``False``)
        If ``True``, then only return the top layer instead of apply the scalar mix.
    max_pieces : int, optional (default: 512)
        The BERT embedder uses positional embeddings and so has a corresponding
        maximum length for its input ids. Assuming the inputs are windowed
        and padded appropriately by this length, the embedder will split them into a
        large batch, feed them into BERT, and recombine the output as if it was a
        longer sequence.
    start_tokens : int, optional (default: 1)
        The number of starting special tokens input to BERT (usually 1, i.e., [CLS])
    end_tokens : int, optional (default: 1)
        The number of ending tokens input to BERT (usually 1, i.e., [SEP])
    combine_layers : str, optional (default: "mix")
        Options: "mix", "last", "all"
    """

    def __init__(
        self,
        bert_model: BertModel,
        max_pieces: int = 512,
        start_tokens: int = 1,
        end_tokens: int = 1,
        layer_dropout: float = 0.0,
        combine_layers: str = "mix",
    ) -> None:
        super().__init__()
        self.bert_model = bert_model
        self.output_dim = bert_model.config.hidden_size
        self.max_pieces = max_pieces
        self.start_tokens = start_tokens
        self.end_tokens = end_tokens
        self.combine_layers = combine_layers

        if self.combine_layers == "mix":
            self._scalar_mix = ScalarMixWithDropout(
                bert_model.config.num_hidden_layers,
                do_layer_norm=False,
                dropout=layer_dropout,
            )
        else:
            self._scalar_mix = None

    def get_output_dim(self) -> int:
        return self.output_dim

    def forward(
        self,
        input_ids: torch.LongTensor,
        offsets: torch.LongTensor = None,
        token_type_ids: torch.LongTensor = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        input_ids : ``torch.LongTensor``
            The (batch_size, ..., max_sequence_length) tensor of wordpiece ids.
        offsets : ``torch.LongTensor``, optional
            The BERT embeddings are one per wordpiece. However it's possible/likely
            you might want one per original token. In that case, ``offsets``
            represents the indices of the desired wordpiece for each original token.
            Depending on how your token indexer is configured, this could be the
            position of the last wordpiece for each token, or it could be the position
            of the first wordpiece for each token.
            For example, if you had the sentence "Definitely not", and if the corresponding
            wordpieces were ["Def", "##in", "##ite", "##ly", "not"], then the input_ids
            would be 5 wordpiece ids, and the "last wordpiece" offsets would be [3, 4].
            If offsets are provided, the returned tensor will contain only the wordpiece
            embeddings at those positions, and (in particular) will contain one embedding
            per token. If offsets are not provided, the entire tensor of wordpiece embeddings
            will be returned.
        token_type_ids : ``torch.LongTensor``, optional
            If an input consists of two sentences (as in the BERT paper),
            tokens from the first sentence should have type 0 and tokens from
            the second sentence should have type 1.  If you don't provide this
            (the default BertIndexer doesn't) then it's assumed to be all 0s.
        """
        # pylint: disable=arguments-differ
        batch_size, full_seq_len = input_ids.size(0), input_ids.size(-1)
        initial_dims = list(input_ids.shape[:-1])

        # The embedder may receive an input tensor that has a sequence length longer than can
        # be fit. In that case, we should expect the wordpiece indexer to create padded windows
        # of length `self.max_pieces` for us, and have them concatenated into one long sequence.
        # E.g., "[CLS] I went to the [SEP] [CLS] to the store to [SEP] ..."
        # We can then split the sequence into sub-sequences of that length, and concatenate them
        # along the batch dimension so we effectively have one huge batch of partial sentences.
        # This can then be fed into BERT without any sentence length issues. Keep in mind
        # that the memory consumption can dramatically increase for large batches with extremely
        # long sentences.
        needs_split = full_seq_len > self.max_pieces
        last_window_size = 0
        if needs_split:
            # Split the flattened list by the window size, `max_pieces`
            split_input_ids = list(input_ids.split(self.max_pieces, dim=-1))

            # We want all sequences to be the same length, so pad the last sequence
            last_window_size = split_input_ids[-1].size(-1)
            padding_amount = self.max_pieces - last_window_size
            split_input_ids[-1] = F.pad(
                split_input_ids[-1], pad=[0, padding_amount], value=0
            )

            # Now combine the sequences along the batch dimension
            input_ids = torch.cat(split_input_ids, dim=0)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        input_mask = (input_ids != 0).long()

        # input_ids may have extra dimensions, so we reshape down to 2-d
        # before calling the BERT model and then reshape back at the end.
        _, _, all_encoder_layers = self.bert_model(
            input_ids=util.combine_initial_dims(input_ids),
            token_type_ids=util.combine_initial_dims(token_type_ids),
            attention_mask=util.combine_initial_dims(input_mask),
        )
        all_encoder_layers = torch.stack(all_encoder_layers[1:])

        if needs_split:
            # First, unpack the output embeddings into one long sequence again
            unpacked_embeddings = torch.split(all_encoder_layers, batch_size, dim=1)
            unpacked_embeddings = torch.cat(unpacked_embeddings, dim=2)

            # Next, select indices of the sequence such that it will result in embeddings representing the original
            # sentence. To capture maximal context, the indices will be the middle part of each embedded window
            # sub-sequence (plus any leftover start and final edge windows), e.g.,
            #  0     1 2    3  4   5    6    7     8     9   10   11   12    13 14  15
            # "[CLS] I went to the very fine [SEP] [CLS] the very fine store to eat [SEP]"
            # with max_pieces = 8 should produce max context indices [2, 3, 4, 10, 11, 12] with additional start
            # and final windows with indices [0, 1] and [14, 15] respectively.

            # Find the stride as half the max pieces, ignoring the special start and end tokens
            # Calculate an offset to extract the centermost embeddings of each window
            stride = (self.max_pieces - self.start_tokens - self.end_tokens) // 2
            stride_offset = stride // 2 + self.start_tokens

            first_window = list(range(stride_offset))

            max_context_windows = [
                i
                for i in range(full_seq_len)
                if stride_offset - 1 < i % self.max_pieces < stride_offset + stride
            ]

            final_window_start = (
                full_seq_len - (full_seq_len % self.max_pieces) + stride_offset + stride
            )
            final_window = list(range(final_window_start, full_seq_len))

            select_indices = first_window + max_context_windows + final_window

            initial_dims.append(len(select_indices))

            recombined_embeddings = unpacked_embeddings[:, :, select_indices]
        else:
            recombined_embeddings = all_encoder_layers

        # Recombine the outputs of all layers
        # (layers, batch_size * d1 * ... * dn, sequence_length, embedding_dim)
        # recombined = torch.cat(combined, dim=2)
        input_mask = (recombined_embeddings != 0).long()

        # At this point, mix is (batch_size * d1 * ... * dn, sequence_length, embedding_dim)

        if offsets is None:
            # Resize to (batch_size, d1, ..., dn, sequence_length, embedding_dim)
            dims = initial_dims if needs_split else input_ids.size()
            layers = util.uncombine_initial_dims(recombined_embeddings, dims)
        else:
            # offsets is (batch_size, d1, ..., dn, orig_sequence_length)
            offsets2d = util.combine_initial_dims(offsets)
            # now offsets is (batch_size * d1 * ... * dn, orig_sequence_length)
            range_vector = util.get_range_vector(
                offsets2d.size(0), device=util.get_device_of(recombined_embeddings)
            ).unsqueeze(1)
            # selected embeddings is also (batch_size * d1 * ... * dn, orig_sequence_length)
            selected_embeddings = recombined_embeddings[:, range_vector, offsets2d]

            layers = util.uncombine_initial_dims(selected_embeddings, offsets.size())

        if self._scalar_mix is not None:
            return self._scalar_mix(layers, input_mask)
        elif self.combine_layers == "last":
            return layers[-1]
        else:
            return layers


@TokenEmbedder.register("udify-bert-pretrained")
class UdifyPretrainedBertEmbedder(BertEmbedder):
    def __init__(
        self,
        pretrained_model: str,
        requires_grad: bool = False,
        dropout: float = 0.1,
        layer_dropout: float = 0.1,
        combine_layers: str = "mix",
    ) -> None:
        model = BertModel.from_pretrained(pretrained_model)

        for param in model.parameters():
            param.requires_grad = requires_grad

        super().__init__(
            bert_model=model, layer_dropout=layer_dropout, combine_layers=combine_layers
        )

        self.model = model
        self.dropout = dropout
        self.set_dropout(dropout)

    def set_dropout(self, dropout):
        """
        Applies dropout to all BERT layers
        """
        self.dropout = dropout

        self.model.embeddings.dropout.p = dropout

        for layer in self.model.encoder.layer:
            layer.attention.self.dropout.p = dropout
            layer.attention.output.dropout.p = dropout
            layer.output.dropout.p = dropout


@TokenEmbedder.register("udify-bert-predictor")
class UdifyPredictionBertEmbedder(BertEmbedder):
    """To be used for inference only, pretrained model is unneeded"""

    def __init__(
        self,
        pretrained_model: str,
        requires_grad: bool = False,
        dropout: float = 0.1,
        layer_dropout: float = 0.1,
        combine_layers: str = "mix",
    ) -> None:
        bert_config = BertConfig.from_pretrained(pretrained_model)
        bert_config.output_hidden_states = True
        model = BertModel(bert_config)

        for param in model.parameters():
            param.requires_grad = requires_grad

        super().__init__(
            bert_model=model, layer_dropout=layer_dropout, combine_layers=combine_layers
        )

        self.model = model
        self.dropout = dropout
        self.set_dropout(dropout)

    def set_dropout(self, dropout):
        """
        Applies dropout to all BERT layers
        """
        self.dropout = dropout

        self.model.embeddings.dropout.p = dropout

        for layer in self.model.encoder.layer:
            layer.attention.self.dropout.p = dropout
            layer.attention.output.dropout.p = dropout
            layer.output.dropout.p = dropout
