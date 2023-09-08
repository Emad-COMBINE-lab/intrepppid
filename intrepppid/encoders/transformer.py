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

import math
from torch import nn
from torch import Tensor
import torch


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class Transformers(nn.Module):
    def __init__(
        self,
        embedding_size,
        num_layers,
        feedforward_size,
        num_heads,
        activation_fn,
        layer_norm,
        dropout_rate,
        trunc_len,
        mean=True,
        truncate_to_longest=True
    ):
        super().__init__()

        self.embedding_size = embedding_size

        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_size,
            nhead=num_heads,
            dim_feedforward=feedforward_size,
            dropout=dropout_rate,
            activation=activation_fn,
            # layer_norm=layer_norm,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            transformer_encoder_layer, num_layers=num_layers
        )

        self.embedding_size = embedding_size
        self.positional_encoder = PositionalEncoding(
            embedding_size, dropout_rate, max_len=trunc_len
        )
        self.nl = nn.Mish()
        self.fc = nn.Linear(self.embedding_size, self.embedding_size)

        self.mean = mean
        self.truncate_to_longest = truncate_to_longest

    def forward(self, x: torch.TensorType):

        padding_mask = (x == torch.zeros(self.embedding_size).to('cuda'))[:, :, 0]

        #x = x.permute(1, 0, 2)

        x = self.positional_encoder(x)

        x = self.transformer_encoder(x)#, src_key_padding_mask=padding_mask)

        #print('x_shape', x.shape)

        if self.mean:
            x = torch.mean(x, dim=1)
        else:
            x = x[:,0,:]

            #print('x_mean', x)

        x = self.nl(x)

        x = self.fc(x)

        return x
