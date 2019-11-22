"""
A modification to AllenNLP's TokenCharactersEncoder
"""

import torch
from allennlp.common import Params
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder
from allennlp.modules.time_distributed import TimeDistributed
from allennlp.modules.token_embedders import TokenEmbedder
from allennlp.modules.token_embedders.embedding import Embedding


@TokenEmbedder.register("udify_character_encoding")
class UdifyTokenCharactersEncoder(TokenEmbedder):
    """
    Like AllenNLP's TokenCharactersEncoder, but applies dropout to the embeddings.
    https://github.com/allenai/allennlp/blob/7dbd7d34a2f1390d1ff01f2e9ed6f8bdaaef77eb/
    allennlp/modules/token_embedders/token_characters_encoder.py
    """

    def __init__(
        self, embedding: Embedding, encoder: Seq2VecEncoder, dropout: float = 0.0
    ) -> None:
        super(UdifyTokenCharactersEncoder, self).__init__()
        self._embedding = TimeDistributed(embedding)
        self._encoder = TimeDistributed(encoder)
        if dropout > 0:
            self._dropout = torch.nn.Dropout(p=dropout)
        else:
            self._dropout = lambda x: x

    def get_output_dim(self) -> int:
        return (
            self._encoder._module.get_output_dim()
        )  # pylint: disable=protected-access

    def forward(
        self, token_characters: torch.Tensor
    ) -> torch.Tensor:  # pylint: disable=arguments-differ
        mask = (token_characters != 0).long()
        return self._encoder(self._dropout(self._embedding(token_characters)), mask)

    # The setdefault requires a custom from_params
    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params):  # type: ignore
        # pylint: disable=arguments-differ
        embedding_params: Params = params.pop("embedding")
        # Embedding.from_params() uses "tokens" as the default namespace, but we need to change
        # that to be "token_characters" by default.
        embedding_params.setdefault("vocab_namespace", "token_characters")
        embedding = Embedding.from_params(vocab, embedding_params)
        encoder_params: Params = params.pop("encoder")
        encoder = Seq2VecEncoder.from_params(encoder_params)
        dropout = params.pop_float("dropout", 0.0)
        params.assert_empty(cls.__name__)
        return cls(embedding, encoder, dropout)
