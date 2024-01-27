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

from torch import nn
from collections import OrderedDict
from intrepppid.utils import WeightDrop
import torch


class AttentionHead(nn.Module):
    def __init__(self, embedding_size, do_rate, num_heads):
        """
        A classifier layer that uses attention.

        :param embedding_size: The size of the embeddings.
        :param do_rate: The dropout rate to use.
        :param num_heads: The number of attention heads to use.
        """
        super().__init__()

        self.embedding_size = embedding_size
        self.do_rate = do_rate
        self.nl = nn.Mish()

        self.classify = nn.Sequential(
            OrderedDict(
                [
                    (
                        "fc1",
                        WeightDrop(
                            nn.Linear(self.embedding_size, self.embedding_size // 2),
                            ["weight"],
                            dropout=self.do_rate,
                            variational=False,
                        ),
                    ),
                    ("nl1", self.nl),
                    ("do1", nn.Dropout(p=self.do_rate)),
                    ("nl2", self.nl),
                    ("do2", nn.Dropout(p=self.do_rate)),
                    (
                        "fc2",
                        WeightDrop(
                            nn.Linear(self.embedding_size // 2, 1),
                            ["weight"],
                            dropout=self.do_rate,
                            variational=False,
                        ),
                    ),
                ]
            )
        )

        self.feed_forward = nn.Sequential(
            OrderedDict(
                [
                    ("nl", self.nl),
                    ("fc", nn.Linear(self.embedding_size, self.embedding_size)),
                    ("nl", self.nl),
                ]
            )
        )

        self.cross_attention = nn.MultiheadAttention(embedding_size, num_heads, batch_first=True)

    def forward(self, x1, x2, padding_x1, padding_x2):

        # Cross-Attention step
        z1, _ = self.cross_attention(x2, x1, x1, key_padding_mask=padding_x1)
        z2, _ = self.cross_attention(x1, x2, x2, key_padding_mask=padding_x2)

        # Mean reduction step
        z1 = self.feed_forward(torch.mean(z1, dim=1))
        z2 = self.feed_forward(torch.mean(z2, dim=1))

        # The rest is the same as the MLP Head
        z = (z1 + z2) / 2

        y_hat = self.classify(z)

        return y_hat
