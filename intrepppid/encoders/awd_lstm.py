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
from intrepppid.utils import WeightDrop


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
