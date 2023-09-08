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
import json
import random
from os import makedirs
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    StochasticWeightAveraging,
)
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from torch import nn
import torchmetrics

from intrepppid.classifier.head import MLPHead
from intrepppid.utils import embedding_dropout, DictLogger, WeightDrop
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim import AdamW
from ranger21 import Ranger21
from typing import Optional, Tuple
from intrepppid.encoders.barlow import BarlowTwinsLoss, make_rnn_barlow_encoder
from intrepppid.data.ppi_oma import IntrepppidDataModule


class BarlowE2ENet(pl.LightningModule):
    def __init__(
        self,
        embedding_size: int,
        batch_size: int,
        encoder: nn.Module,
        head: nn.Module,
        embedding_droprate: float,
        num_epochs: int,
        steps_per_epoch: int,
        beta_classifier: float,
        use_projection: bool,
        projection_dropconnect: float,
        optimizer_type: str,
        lr: float,
    ):
        super().__init__()
        self.encoder = encoder
        self.embedding_droprate = embedding_droprate
        self.classifier_criterion = nn.BCEWithLogitsLoss()
        self.num_epochs = num_epochs
        self.steps_per_epoch = steps_per_epoch
        self.use_projection = use_projection
        self.lr = lr

        if optimizer_type in ["ranger21", "adamw"]:
            self.optimizer_type = optimizer_type
        else:
            raise ValueError(
                'Expected one of "ranger21" or "adamw" as the optimizer type.'
            )

        self.barlow_criterion = BarlowTwinsLoss(batch_size, z_dim=embedding_size)

        if self.use_projection:
            self.barlow_projection = nn.Sequential(
                nn.Mish(),
                WeightDrop(
                    nn.Linear(embedding_size, embedding_size),
                    ["weight"],
                    projection_dropconnect,
                    False,
                ),
            )

        self.auroc = torchmetrics.AUROC(task="binary")
        self.average_precision = torchmetrics.AveragePrecision(task="binary")
        self.mcc = torchmetrics.MatthewsCorrCoef(task="binary", threshold=0.5)
        self.precision_metric = torchmetrics.Precision(task="binary")
        self.recall = torchmetrics.Recall(task="binary")

        self.do_rate = 0.3
        self.head = head
        self.beta_classifier = beta_classifier

    def embedding_dropout(self, embed, words, p=0.2):
        return embedding_dropout(self.training, embed, words, p)

    def forward(self, x1, x2, freeze_encoder: bool = False, freeze_head: bool = False):

        if freeze_encoder:
            with torch.no_grad():
                z1 = self.encoder(x1)
                z2 = self.encoder(x2)
        else:
            z1 = self.encoder(x1)
            z2 = self.encoder(x2)

        if freeze_head:
            with torch.no_grad():
                y_hat = self.head(z1, z2)
        else:
            y_hat = self.head(z1, z2)

        return y_hat

    def step(self, batch, stage):
        p1_seq, p2_seq, omid1_seq, omid2_seq, y = batch

        z_omid1 = self.encoder(omid1_seq)
        z_omid2 = self.encoder(omid1_seq)

        # If there are three dimensions to the output, then this is a Transformer encoder result that hasn't been
        # averaged. We'll do that here before we project and call the barlow criterion.
        if len(z_omid1.shape) == 3:
            z_omid1 = torch.mean(z_omid1, dim=0)
            z_omid2 = torch.mean(z_omid2, dim=0)

        if self.use_projection:
            z_omid1 = self.barlow_projection(z_omid1)
            z_omid2 = self.barlow_projection(z_omid2)

        # We flatten for when we don't average the Transformer encoder result for use with the AttentionHead
        # z_omid1 = torch.flatten(z_omid1.permute(1, 0, 2), start_dim=1)
        # z_omid2 = torch.flatten(z_omid2.permute(1, 0, 2), start_dim=1)

        barlow_loss = self.barlow_criterion(z_omid1, z_omid2)

        y_hat = self(p1_seq, p2_seq).squeeze(1)

        classifier_loss = self.classifier_criterion(y_hat, y.float())

        norm_beta_ssl = 1 / self.beta_classifier
        norm_beta_classifier = 1 - norm_beta_ssl

        loss = norm_beta_classifier * classifier_loss + norm_beta_ssl * barlow_loss
        #loss = classifier_loss + barlow_loss

        auroc = self.auroc(y_hat, y)
        ap = self.average_precision(y_hat, y)
        mcc = self.mcc(y_hat, y)
        pr = self.precision_metric(y_hat, y)
        rec = self.recall(y_hat, y)

        self.log(
            f"{stage}_classifier_loss",
            classifier_loss,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )
        self.log(
            f"{stage}_barlow_loss",
            barlow_loss,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )
        self.log(f"{stage}_loss", loss, on_epoch=True, on_step=False, prog_bar=True)

        self.log(
            f"{stage}_classifier_loss_step",
            classifier_loss,
            on_epoch=False,
            on_step=True,
            prog_bar=False,
        )
        self.log(
            f"{stage}_barlow_loss_step",
            barlow_loss,
            on_epoch=False,
            on_step=True,
            prog_bar=False,
        )
        self.log(
            f"{stage}_loss_step", loss, on_epoch=False, on_step=True, prog_bar=False
        )

        self.log(f"{stage}_auroc", auroc, on_epoch=True, on_step=False)
        self.log(f"{stage}_ap", ap, on_epoch=True, on_step=False)
        self.log(f"{stage}_mcc", mcc, on_epoch=True, on_step=False)
        self.log(f"{stage}_precision", pr, on_epoch=True, on_step=False)
        self.log(f"{stage}_rec", rec, on_epoch=True, on_step=False)

        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, "test")

    def configure_optimizers(self):
        if self.optimizer_type == "ranger21":
            optimizer = Ranger21(
                self.parameters(),
                use_warmup=False,
                warmdown_active=False,
                lr=self.lr,
                weight_decay=1e-2,
                num_batches_per_epoch=self.steps_per_epoch,
                num_epochs=self.num_epochs,
                warmdown_start_pct=0.72,
            )
        elif self.optimizer_type == "adamw":
            optimizer = AdamW(self.parameters(), lr=1e-3)
        else:
            raise ValueError(
                'Expected one of "ranger21" or "adamw" as the optimizer type.'
            )
        # scheduler = OneCycleLR(optimizer, 0.01, epochs=self.num_epochs, steps_per_epoch=self.steps_per_epoch)

        # return [optimizer], [scheduler]
        return optimizer


def train_e2e_rnn_barlow(
    vocab_size: int,
    trunc_len: int,
    embedding_size: int,
    rnn_num_layers: int,
    rnn_dropout_rate: float,
    variational_dropout: bool,
    bi_reduce: str,
    ppi_dataset_path: Path,
    sentencepiece_path: Path,
    log_path: Path,
    hyperparams_path: Path,
    chkpt_dir: Path,
    c_type: int,
    model_name: str,
    workers: int,
    embedding_droprate: float,
    do_rate: float,
    num_epochs: int,
    batch_size: int,
    encoder_only_steps: int,
    classifier_warm_up: int,
    beta_classifier: float,
    use_projection: bool,
    projection_dropconnect: float,
    optimizer_type: str,
    lr: float,
    seed: Optional[int] = None,
):
    makedirs(chkpt_dir, exist_ok=True)
    makedirs(log_path, exist_ok=True)
    makedirs(hyperparams_path.parent, exist_ok=True)

    seed = random.randint(0, 99999) if seed is None else seed

    seed_everything(seed)

    hyperparameters = {
        "architecture": "ClassifierBarlow",
        "vocab_size": vocab_size,
        "trunc_len": trunc_len,
        "embedding_size": embedding_size,
        "rnn_num_layers": rnn_num_layers,
        "rnn_dropout_rate": rnn_dropout_rate,
        "variational_dropout": variational_dropout,
        "bi_reduce": bi_reduce,
        "ppi_dataset_path": str(ppi_dataset_path),
        "sentencepiece_path": str(sentencepiece_path),
        "log_path": str(log_path),
        "hyperparams_path": str(hyperparams_path),
        "chkpt_dir": str(chkpt_dir),
        "model_name": model_name,
        "workers": workers,
        "embedding_droprate": embedding_droprate,
        "do_rate": do_rate,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "encoder_only_steps": encoder_only_steps,
        "classifier_warm_up": classifier_warm_up,
        "beta_classifier": beta_classifier,
        "use_projection": use_projection,
        "projection_dropconnect": projection_dropconnect,
        "optimizer_type": optimizer_type,
        "lr": lr,
        "seed": seed,
    }

    with open(hyperparams_path, "w") as f:
        json.dump(hyperparameters, f)

    data_module = IntrepppidDataModule(
        batch_size=batch_size,
        dataset_path=ppi_dataset_path,
        c_type=c_type,
        trunc_len=trunc_len,
        workers=workers,
        vocab_size=vocab_size,
        model_file=sentencepiece_path,
        seed=seed,
        sos=False,
        eos=False
    )

    data_module.setup("training")
    steps_per_epoch = len(data_module.train_dataloader())

    encoder = make_rnn_barlow_encoder(
        vocab_size,
        embedding_size,
        rnn_num_layers,
        rnn_dropout_rate,
        variational_dropout,
        bi_reduce,
        batch_size,
        embedding_droprate,
        num_epochs,
        steps_per_epoch,
    )

    head = MLPHead(embedding_size, do_rate)

    net = BarlowE2ENet(
        embedding_size,
        batch_size,
        encoder,
        head,
        embedding_droprate,
        num_epochs,
        steps_per_epoch,
        beta_classifier,
        use_projection,
        projection_dropconnect,
        optimizer_type,
        lr,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=chkpt_dir,
        filename=model_name + "-{epoch:02d}-{val_loss:.2f}",
    )

    dict_logger = DictLogger()
    tb_logger = TensorBoardLogger(f"{log_path}", name="tensorboard", version=model_name)
    lr_monitor = LearningRateMonitor(logging_interval="step")
    swa = StochasticWeightAveraging(swa_lrs=1e-2)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=num_epochs,
        precision=16,
        logger=[dict_logger, tb_logger],
        callbacks=[checkpoint_callback, lr_monitor, swa],
        log_every_n_steps=2,
    )

    trainer.fit(net, data_module)

    test_results = trainer.test(dataloaders=data_module, ckpt_path="best")

    dict_logger.metrics["test_results"] = test_results

    with open(log_path / model_name / "metrics.json", "w") as f:
        json.dump(dict_logger.metrics, f, indent=3)
