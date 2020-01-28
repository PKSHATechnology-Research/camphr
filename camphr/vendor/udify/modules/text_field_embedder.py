"""
A modification to AllenNLP's TextFieldEmbedder
"""

import warnings
from typing import Dict, List, Optional

import torch
from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.time_distributed import TimeDistributed
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder
from torch.nn.modules.linear import Linear


@TextFieldEmbedder.register("udify_embedder")
class UdifyTextFieldEmbedder(TextFieldEmbedder):
    """
    This is a ``TextFieldEmbedder`` that, instead of concatenating embeddings together
    as in ``BasicTextFieldEmbedder``, sums them together. It also optionally allows
    dropout to be applied to its output. See AllenNLP's basic_text_field_embedder.py,
    https://github.com/allenai/allennlp/blob/a75cb0a9ddedb23801db74629c3a3017dafb375e/
    allennlp/modules/text_field_embedders/
    basic_text_field_embedder.py
    """

    def __init__(
        self,
        token_embedders: Dict[str, TokenEmbedder],
        output_dim: Optional[int] = None,
        sum_embeddings: List[str] = None,
        embedder_to_indexer_map: Dict[str, List[str]] = None,
        allow_unmatched_keys: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super(UdifyTextFieldEmbedder, self).__init__()
        self._output_dim = output_dim
        self._token_embedders = token_embedders
        self._embedder_to_indexer_map = embedder_to_indexer_map
        for key, embedder in token_embedders.items():
            name = "token_embedder_%s" % key
            self.add_module(name, embedder)
        self._allow_unmatched_keys = allow_unmatched_keys
        self._dropout = torch.nn.Dropout(p=dropout) if dropout > 0 else lambda x: x
        self._sum_embeddings = sum_embeddings if sum_embeddings is not None else []

        hidden_dim = 0
        for embedder in self._token_embedders.values():
            hidden_dim += embedder.get_output_dim()

        if len(self._sum_embeddings) > 1:
            for key in self._sum_embeddings[1:]:
                hidden_dim -= self._token_embedders[key].get_output_dim()

        if self._output_dim is None:
            self._projection_layer = None
            self._output_dim = hidden_dim
        else:
            self._projection_layer = Linear(hidden_dim, self._output_dim)

    def get_output_dim(self) -> int:
        return self._output_dim

    def forward(
        self, text_field_input: Dict[str, torch.Tensor], num_wrapping_dims: int = 0
    ) -> torch.Tensor:
        embedder_keys = self._token_embedders.keys()
        input_keys = text_field_input.keys()

        # Check for unmatched keys
        if not self._allow_unmatched_keys:
            if embedder_keys < input_keys:
                # token embedder keys are a strict subset of text field input keys.
                message = (
                    f"Your text field is generating more keys ({list(input_keys)}) "
                    f"than you have token embedders ({list(embedder_keys)}. "
                    f"If you are using a token embedder that requires multiple keys "
                    f"(for example, the OpenAI Transformer embedder or the BERT embedder) "
                    f"you need to add allow_unmatched_keys = True "
                    f"(and likely an embedder_to_indexer_map) to your "
                    f"BasicTextFieldEmbedder configuration. "
                    f"Otherwise, you should check that there is a 1:1 embedding "
                    f"between your token indexers and token embedders."
                )
                raise ConfigurationError(message)

            elif self._token_embedders.keys() != text_field_input.keys():
                # some other mismatch
                message = "Mismatched token keys: %s and %s" % (
                    str(self._token_embedders.keys()),
                    str(text_field_input.keys()),
                )
                raise ConfigurationError(message)

        def embed(key):
            # If we pre-specified a mapping explictly, use that.
            if self._embedder_to_indexer_map is not None:
                tensors = [
                    text_field_input[indexer_key]
                    for indexer_key in self._embedder_to_indexer_map[key]
                ]
            else:
                # otherwise, we assume the mapping between indexers and embedders
                # is bijective and just use the key directly.
                tensors = [text_field_input[key]]
            # Note: need to use getattr here so that the pytorch voodoo
            # with submodules works with multiple GPUs.
            embedder = getattr(self, "token_embedder_{}".format(key))
            for _ in range(num_wrapping_dims):
                embedder = TimeDistributed(embedder)
            token_vectors = embedder(*tensors)

            return token_vectors

        embedded_representations = []
        keys = sorted(embedder_keys)

        sum_embed = []
        for key in self._sum_embeddings:
            token_vectors = embed(key)
            sum_embed.append(token_vectors)
            keys.remove(key)

        if sum_embed:
            embedded_representations.append(sum(sum_embed))

        for key in keys:
            token_vectors = embed(key)
            embedded_representations.append(token_vectors)

        combined_embeddings = self._dropout(torch.cat(embedded_representations, dim=-1))

        if self._projection_layer is not None:
            combined_embeddings = self._dropout(
                self._projection_layer(combined_embeddings)
            )

        return combined_embeddings

    # This is some unusual logic, it needs a custom from_params.
    @classmethod
    def from_params(
        cls, vocab: Vocabulary, params: Params
    ) -> "UdifyTextFieldEmbedder":  # type: ignore
        # pylint: disable=arguments-differ,bad-super-call

        # The original `from_params` for this class was designed in a way that didn't agree
        # with the constructor. The constructor wants a 'token_embedders' parameter that is a
        # `Dict[str, TokenEmbedder]`, but the original `from_params` implementation expected those
        # key-value pairs to be top-level in the params object.
        #
        # This breaks our 'configuration wizard' and configuration checks. Hence, going forward,
        # the params need a 'token_embedders' key so that they line up with what the constructor wants.
        # For now, the old behavior is still supported, but produces a DeprecationWarning.

        embedder_to_indexer_map = params.pop("embedder_to_indexer_map", None)
        if embedder_to_indexer_map is not None:
            embedder_to_indexer_map = embedder_to_indexer_map.as_dict(quiet=True)
        allow_unmatched_keys = params.pop_bool("allow_unmatched_keys", False)

        token_embedder_params = params.pop("token_embedders", None)

        dropout = params.pop_float("dropout", 0.0)

        output_dim = params.pop_int("output_dim", None)
        sum_embeddings = params.pop("sum_embeddings", None)

        if token_embedder_params is not None:
            # New way: explicitly specified, so use it.
            token_embedders = {
                name: TokenEmbedder.from_params(subparams, vocab=vocab)
                for name, subparams in token_embedder_params.items()
            }

        else:
            # Warn that the original behavior is deprecated
            warnings.warn(
                DeprecationWarning(
                    "the token embedders for BasicTextFieldEmbedder should now "
                    "be specified as a dict under the 'token_embedders' key, "
                    "not as top-level key-value pairs"
                )
            )

            token_embedders = {}
            keys = list(params.keys())
            for key in keys:
                embedder_params = params.pop(key)
                token_embedders[key] = TokenEmbedder.from_params(
                    vocab=vocab, params=embedder_params
                )

        params.assert_empty(cls.__name__)
        return cls(
            token_embedders,
            output_dim,
            sum_embeddings,
            embedder_to_indexer_map,
            allow_unmatched_keys,
            dropout,
        )
