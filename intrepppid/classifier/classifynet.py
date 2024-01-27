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

import pytorch_lightning as pl
from torch import nn
import torchmetrics
from intrepppid.utils import embedding_dropout
import torch
from ranger21 import Ranger21
from typing import Optional


class ClassifyNet(pl.LightningModule):
    def __init__(
        self,
        encoder,
        head,
        embedding_droprate: float,
        num_epochs: int,
        steps_per_epoch: int,
        fine_tune_epochs: Optional[int] = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.embedding_droprate = embedding_droprate
        self.criterion = nn.BCEWithLogitsLoss()
        self.num_epochs = num_epochs
        self.steps_per_epoch = steps_per_epoch
        self.fine_tune_epochs = fine_tune_epochs

        self.auroc = torchmetrics.AUROC(task="binary")
        self.average_precision = torchmetrics.AveragePrecision(task="binary")
        self.mcc = torchmetrics.MatthewsCorrCoef(task="binary", threshold=0.5)
        self.precision_metric = torchmetrics.Precision(task="binary")
        self.recall = torchmetrics.Recall(task="binary")

        self.do_rate = 0.3
        self.head = head

    def embedding_dropout(self, embed, words, p=0.2):
        return embedding_dropout(self.training, embed, words, p)

    def forward(self, x1, x2):
        if (
            self.fine_tune_epochs is not None
            and self.current_epoch >= self.num_epochs - self.fine_tune_epochs
        ):
            x1 = self.encoder(x1)
            x2 = self.encoder(x2)
        else:
            with torch.no_grad():
                x1 = self.encoder(x1)
                x2 = self.encoder(x2)

        y_hat = self.head(x1, x2)

        return y_hat

    def step(self, batch, stage):
        x1, x2, y = batch

        y_hat = self(x1, x2).squeeze(1)

        loss = self.criterion(y_hat, y.float())

        self.log(f"{stage}_loss", loss, on_epoch=True, on_step=False, prog_bar=True)

        auroc = self.auroc(y_hat, y)
        self.log(f"{stage}_auroc", auroc, on_epoch=True, on_step=False)

        ap = self.average_precision(y_hat, y)
        self.log(f"{stage}_ap", ap, on_epoch=True, on_step=False)

        mcc = self.mcc(y_hat, y)
        self.log(f"{stage}_mcc", mcc, on_epoch=True, on_step=False)

        pr = self.precision_metric(y_hat, y)
        self.log(f"{stage}_precision", pr, on_epoch=True, on_step=False)

        rec = self.recall(y_hat, y)
        self.log(f"{stage}_rec", rec, on_epoch=True, on_step=False)

        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, "test")

    def configure_optimizers(self):
        optimizer = Ranger21(
            self.parameters(),
            lr=1e-3,
            weight_decay=1e-4,
            num_batches_per_epoch=self.steps_per_epoch,
            num_epochs=self.num_epochs,
            warmdown_start_pct=0.72,
        )
        return optimizer
