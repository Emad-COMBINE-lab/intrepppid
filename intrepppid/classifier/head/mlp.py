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


class MLPHead(nn.Module):
    def __init__(self, embedding_size, do_rate):
        """
        A multilayer perceptron (MLP) classifier layer.

        :param embedding_size: The size of the embeddings.
        :param do_rate: The dropout rate to use.
        """
        super().__init__()

        self.embedding_size = embedding_size
        self.do_rate = do_rate

        self.classify = nn.Sequential(
            OrderedDict(
                [
                    ("nl0", nn.Mish()),
                    (
                        "fc1",
                        WeightDrop(
                            nn.Linear(self.embedding_size, self.embedding_size // 2),
                            ["weight"],
                            dropout=self.do_rate,
                            variational=False,
                        ),
                    ),
                    ("nl1", nn.Mish()),
                    ("do1", nn.Dropout(p=self.do_rate)),
                    ("nl2", nn.Mish()),
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

    def forward(self, x1, x2):
        x = (x1 + x2) / 2

        return self.classify(x)
