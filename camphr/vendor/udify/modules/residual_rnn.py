"""
Defines a "Residual RNN", which adds the input of the RNNs to the output, which tends to work better than conventional
RNN layers.
"""

import torch
from allennlp.common.checks import ConfigurationError
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from overrides import overrides


@Seq2SeqEncoder.register("udify_residual_rnn")
class ResidualRNN(Seq2SeqEncoder):
    """
    Instead of using AllenNLP's default PyTorch wrapper for seq2seq layers, we would like
    to apply intermediate logic between each layer, so we create a new wrapper for each layer,
    with residual connections between.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        residual: bool = True,
        rnn_type: str = "lstm",
    ) -> None:
        super(ResidualRNN, self).__init__()

        self._input_size = input_size
        self._hidden_size = hidden_size
        self._dropout = torch.nn.Dropout(p=dropout)
        self._residual = residual

        rnn_type = rnn_type.lower()
        if rnn_type == "lstm":
            rnn_cell = torch.nn.LSTM
        elif rnn_type == "gru":
            rnn_cell = torch.nn.GRU
        else:
            raise ConfigurationError(f"Unknown RNN cell type {rnn_type}")

        layers = []
        for layer_index in range(num_layers):
            # Use hidden size on later layers so that the first layer projects and all other layers are residual
            input_ = input_size if layer_index == 0 else hidden_size
            rnn = rnn_cell(input_, hidden_size, bidirectional=True, batch_first=True)
            layer = PytorchSeq2SeqWrapper(rnn)
            layers.append(layer)
            self.add_module("rnn_layer_{}".format(layer_index), layer)
        self._layers = layers

    @overrides
    def get_input_dim(self) -> int:
        return self._input_size

    @overrides
    def get_output_dim(self) -> int:
        return self._hidden_size

    @overrides
    def is_bidirectional(self):
        return True

    def forward(
        self, inputs: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:  # pylint: disable=arguments-differ
        hidden = inputs
        for i, layer in enumerate(self._layers):
            encoded = layer(hidden, mask)
            # Sum the backward and forward states to allow residual connections
            encoded = (
                encoded[:, :, : self._hidden_size] + encoded[:, :, self._hidden_size :]
            )

            projecting = i == 0 and self._input_size != self._hidden_size
            if self._residual and not projecting:
                hidden = hidden + self._dropout(encoded)
            else:
                hidden = self._dropout(encoded)

        return hidden
