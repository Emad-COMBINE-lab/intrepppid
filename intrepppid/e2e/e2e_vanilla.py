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

from intrepppid.classifier.head import MLPHead, AttentionHead
from intrepppid.utils import embedding_dropout, DictLogger, WeightDrop
from torch.optim import AdamW
from ranger21 import Ranger21
from typing import Optional
from intrepppid.encoders.barlow import BarlowTwinsLoss, make_rnn_barlow_encoder, make_transformers_barlow_encoder
from intrepppid.data.ppi_oma import IntrepppidDataModule


class VanillaE2ENet(pl.LightningModule):
    def __init__(
        self,
        encoder: nn.Module,
        head: nn.Module,
        embedding_droprate: float,
        num_epochs: int,
        steps_per_epoch: int,
        optimizer_type: str,
        lr: float,
        head_takes_padding: bool = False
    ):
        super().__init__()
        self.encoder = encoder
        self.embedding_droprate = embedding_droprate
        self.classifier_criterion = nn.BCEWithLogitsLoss()
        self.num_epochs = num_epochs
        self.steps_per_epoch = steps_per_epoch
        self.lr = lr

        if optimizer_type in ["ranger21", "adamw"]:
            self.optimizer_type = optimizer_type
        else:
            raise ValueError(
                'Expected one of "ranger21" or "adamw" as the optimizer type.'
            )

        self.auroc = torchmetrics.AUROC(task="binary")
        self.average_precision = torchmetrics.AveragePrecision(task="binary")
        self.mcc = torchmetrics.MatthewsCorrCoef(task="binary", threshold=0.5)
        self.precision_metric = torchmetrics.Precision(task="binary")
        self.recall = torchmetrics.Recall(task="binary")

        self.do_rate = 0.3
        self.head = head

        self.head_takes_padding = head_takes_padding

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

        if self.head_takes_padding:
            # Ok, this is about to fly out of my head the moment I walk away from this keyboard so here's the low-down.
            # x1 is padded to 1000 tokens. the encoder will trim that to the longest sequence in the batch.
            # this means that x1 will have 1000 tokens, and z1 will have <=1000 tokens.
            # we need the padding mask to be the same length as z1, so we'll slice padding_mask_x1 such that it's the
            # same size as z1.
            padding_mask_x1 = x1 == 0
            padding_mask_x1 = padding_mask_x1[:, :z1.shape[1]]

            padding_mask_x2 = x2 == 0
            padding_mask_x2 = padding_mask_x2[:, :z2.shape[1]]

        if freeze_head:
            with torch.no_grad():
                if self.head_takes_padding:
                    y_hat = self.head(z1, z2, padding_mask_x1, padding_mask_x2)
                else:
                    y_hat = self.head(z1, z2)
        else:
            if self.head_takes_padding:
                y_hat = self.head(z1, z2, padding_mask_x1, padding_mask_x2)
            else:
                y_hat = self.head(z1, z2)

        return y_hat

    def step(self, batch, stage):
        p1_seq, p2_seq, omid1_seq, omid2_seq, y = batch

        y_hat = self(p1_seq, p2_seq)
        y_hat = y_hat.squeeze(1)

        loss = self.classifier_criterion(y_hat, y.float())

        auroc = self.auroc(y_hat, y)
        ap = self.average_precision(y_hat, y)
        mcc = self.mcc(y_hat, y)
        pr = self.precision_metric(y_hat, y)
        rec = self.recall(y_hat, y)

        self.log(
            f"{stage}_loss",
            loss,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
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


def train_e2e_transformer_vanilla_attention(
    vocab_size: int,
    trunc_len: int,
    embedding_size: int,
    transformer_num_layers: int,
    transformer_feedforward_size: int,
    transformer_num_heads: int,
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
    optimizer_type: str,
    lr: float,
    cross_attention_heads: int,
    gpu: int = 0,
    seed: Optional[int] = None,
):
    makedirs(chkpt_dir, exist_ok=True)
    makedirs(log_path, exist_ok=True)
    makedirs(hyperparams_path.parent, exist_ok=True)

    seed = random.randint(0, 99999) if seed is None else seed

    seed_everything(seed)

    hyperparameters = {
        "architecture": "E2ETransformerVanillaAttention",
        "vocab_size": vocab_size,
        "trunc_len": trunc_len,
        "embedding_size": embedding_size,
        "transformer_num_layers": transformer_num_layers,
        "transformer_feedforward_size": transformer_feedforward_size,
        "transformer_num_heads": transformer_num_heads,
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
        "optimizer_type": optimizer_type,
        "lr": lr,
        "cross_attention_heads": cross_attention_heads,
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
        eos=True,
        sos=True,
        negative_omid=False
    )

    data_module.setup("training")
    steps_per_epoch = len(data_module.train_dataloader())

    encoder = make_transformers_barlow_encoder(
        vocab_size,
        embedding_size,
        transformer_num_layers,
        do_rate,
        transformer_feedforward_size,
        transformer_num_heads,
        nn.Mish(),
        1e-5,
        batch_size,
        embedding_droprate,
        num_epochs,
        steps_per_epoch,
        trunc_len,
        mean=False,
        truncate_to_longest=True
    )

    head = AttentionHead(embedding_size, do_rate, cross_attention_heads)

    net = VanillaE2ENet(
        encoder,
        head,
        embedding_droprate,
        num_epochs,
        steps_per_epoch,
        optimizer_type,
        lr,
        head_takes_padding=True
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
        devices=[gpu],
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
