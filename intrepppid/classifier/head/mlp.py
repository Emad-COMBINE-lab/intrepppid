from torch import nn
from collections import OrderedDict
from intrepppid.utils import WeightDrop


class MLPHead(nn.Module):
    def __init__(self, embedding_size, do_rate):
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
