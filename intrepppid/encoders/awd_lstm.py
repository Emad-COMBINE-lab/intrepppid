# This file is part of the INTREPPPID programme.
# Copyright (c) 2023 Joseph Szymborski.
#
# This programme is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, version 3.
#
# This programme is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public
# License along with this programme. If not, see
# <https://www.gnu.org/licenses/agpl-3.0.en.html>.

import torch
from torch import nn

from intrepppid.utils import WeightDrop, embedding_dropout


class AWDLSTM(nn.Module):
    def __init__(
        self,
        embedding_size,
        rnn_num_layers,
        lstm_dropout_rate,
        variational_dropout,
        bi_reduce,
    ):
        super().__init__()
        self.bi_reduce = bi_reduce

        self.rnn = nn.LSTM(
            embedding_size,
            embedding_size,
            rnn_num_layers,
            bidirectional=True,
            batch_first=True,
        )

        self.rnn_dp = WeightDrop(
            self.rnn, ["weight_hh_l0"], lstm_dropout_rate, variational_dropout
        )

        self.fc = nn.Linear(embedding_size, embedding_size)
        self.nl = nn.Mish()
        self.embedding_size = embedding_size

    def forward(self, x):
        # Truncate to longest sequence in batch
        max_len = torch.max(torch.sum(x != 0, axis=1))
        x = x[:, :max_len]

        x, (hn, cn) = self.rnn_dp(x)

        if self.bi_reduce == "concat":
            # Concat both directions
            x = hn[-2:, :, :].permute(1, 0, 2).flatten(start_dim=1)
        elif self.bi_reduce == "max":
            # Max both directions
            x = torch.max(hn[-2:, :, :], dim=0).values
        elif self.bi_reduce == "mean":
            # Mean both directions
            x = torch.mean(hn[-2:, :, :], dim=0)
        elif self.bi_reduce == "last":
            # Just use last direction
            x = hn[-1:, :, :].squeeze(0)

        x = self.fc(x)
        # x = self.nl(x)

        return x


class Projection(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers):
        """
        This module implements a simple MLP which is used as a projection layer.

        :param in_dim: The size of the input vector
        :param out_dim: The size of the output vector
        :param num_layers: The total number of layers
        """
        super().__init__()

        diff_dim = (out_dim - in_dim) // num_layers

        layers = []

        dim = in_dim

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(dim, dim + diff_dim))
            layers.append(nn.ReLU())

            dim += diff_dim

        layers.append(nn.Linear(dim, out_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class AWDLSTMEncoder(nn.Module):
    def __init__(
        self,
        embedder: nn.Module,
        embedding_size: int,
        embedding_droprate: float,
        rnn_num_layers: int,
        rnn_dropout_rate: float,
        variational_dropout: bool,
        bi_reduce: str
    ):
        """
        Represents an AWD-LSTM encoder.

        :param embedder: The object responsible for embedding sequences. Must follow the nn.Embedding API.
        :param embedding_size: The nn.Module responsible for encoding embedded sequences.
        :param embedding_droprate: The drop-rate to use for the Embedding Dropout (a la AWD-LSTM).
        :param rnn_num_layers: The number of layers in the AWD-LSTM.
        :param rnn_dropout_rate: The drop-connect rate for the AWD-LSTM.
        :param variational_dropout: Whether the LSTM should use variational dropout (True) or not (False).
        :param bi_reduce: Method to reduce the two LSTM embeddings for both directions. Must be one of "concat", "max", "mean", "last"
        """
        super().__init__()
        self.embedder = embedder
        self.embedding_droprate = embedding_droprate
        self.encoder = AWDLSTM(
            embedding_size,
            rnn_num_layers,
            rnn_dropout_rate,
            variational_dropout,
            bi_reduce
        )
        self.projection = Projection(
            self.encoder.embedding_size, self.encoder.embedding_size * 2, 3
        )

    def embedding_dropout(self, embed, words, p=0.2):
        return embedding_dropout(self.training, embed, words, p)

    def forward(self, x):
        # Truncate to the longest sequence in batch
        max_len = torch.max(torch.sum(x != 0, axis=1))
        x = x[:, :max_len]

        x = self.embedding_dropout(self.embedder, x, p=self.embedding_droprate)
        x = self.encoder(x)

        return x
