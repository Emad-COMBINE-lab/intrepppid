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
import torch.nn.functional as F
from ranger21 import Ranger21
import pytorch_lightning as pl
from intrepppid.utils import DictLogger
from intrepppid.data import OmaTripletDataModule
from pathlib import Path
from intrepppid.encoders import AWDLSTM, Transformers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.seed import seed_everything
import json
from os import makedirs
from typing import Union, Callable
from intrepppid.utils import embedding_dropout


class BarlowTwinsLoss(nn.Module):
    def __init__(self, batch_size, lambda_coeff=5e-3, z_dim=128, eps=1e-5):
        super().__init__()

        self.z_dim = z_dim
        self.batch_size = batch_size
        self.lambda_coeff = lambda_coeff
        self.eps = eps  # Epsilon will help avoid divide-by-zero errors

    def off_diagonal_ele(self, x):
        # taken from: https://github.com/facebookresearch/barlowtwins/blob/main/main.py
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, z1, z2):
        # N x D, where N is the batch size and D is output dim of projection head
        z1_norm = (z1 - torch.mean(z1, dim=0)) / (torch.std(z1, dim=0) + self.eps)
        z2_norm = (z2 - torch.mean(z2, dim=0)) / (torch.std(z2, dim=0) + self.eps)

        cross_corr = torch.matmul(z1_norm.T, z2_norm) / self.batch_size

        on_diag = torch.diagonal(cross_corr).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal_ele(cross_corr).pow_(2).sum()

        return (on_diag + self.lambda_coeff * off_diag) / self.z_dim


class Projection(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers):
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


class BarlowEncoder(pl.LightningModule):
    def __init__(
        self,
        batch_size,
        embedder,
        encoder,
        embedding_droprate,
        num_epochs,
        steps_per_epoch,
    ):
        super().__init__()
        self.embedder = embedder
        self.embedding_droprate = embedding_droprate
        self.encoder = encoder
        self.loss_fn = BarlowTwinsLoss(
            batch_size, z_dim=self.encoder.embedding_size * 2
        )
        self.num_epochs = num_epochs
        self.steps_per_epoch = steps_per_epoch
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

    def step(self, batch, stage):
        x_anchor, x_positive, x_negative = batch

        z_anchor = self.projection(F.relu(self(x_anchor)))
        z_positive = self.projection(F.relu(self(x_positive)))

        loss = self.loss_fn(z_anchor, z_positive)

        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{stage}_loss_step", loss, on_step=True, on_epoch=False)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, "train")

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, "val")

        return loss

    def test_step(self, batch, batch_idx):
        loss = self.step(batch, "test")

        return loss

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        optimizer = Ranger21(
            self.parameters(),
            lr=1e-3,
            weight_decay=1e-4,
            num_batches_per_epoch=self.steps_per_epoch,
            num_epochs=self.num_epochs,
            warmdown_start_pct=0.72,
        )
        return optimizer


def make_rnn_barlow_encoder(
    vocab_size: int,
    embedding_size: int,
    rnn_num_layers: int,
    rnn_dropout_rate: float,
    variational_dropout: bool,
    bi_reduce: str,
    batch_size: int,
    embedding_droprate: float,
    num_epochs: int,
    steps_per_epoch: int,
):
    embedder = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
    encoder = AWDLSTM(
        embedding_size, rnn_num_layers, rnn_dropout_rate, variational_dropout, bi_reduce
    )
    model = BarlowEncoder(
        batch_size, embedder, encoder, embedding_droprate, num_epochs, steps_per_epoch
    )

    return model


def make_transformers_barlow_encoder(
    vocab_size: int,
    embedding_size: int,
    num_layers: int,
    dropout_rate: float,
    feedforward_size: int,
    num_heads: int,
    activation_fn: Union[str, Callable],
    layer_norm: float,
    batch_size: int,
    embedding_droprate: float,
    num_epochs: int,
    steps_per_epoch: int,
    trunc_len: int,
    mean: bool = True,
    truncate_to_longest: bool = True
):
    embedder = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
    encoder = Transformers(
        embedding_size=embedding_size,
        num_layers=num_layers,
        feedforward_size=feedforward_size,
        dropout_rate=dropout_rate,
        num_heads=num_heads,
        activation_fn=activation_fn,
        layer_norm=layer_norm,
        trunc_len=trunc_len,
        mean=mean,
        truncate_to_longest=truncate_to_longest
    )
    model = BarlowEncoder(
        batch_size, embedder, encoder, embedding_droprate, num_epochs, steps_per_epoch
    )

    return model


def train_rnn(
    batch_size: int,
    dataset_path: Path,
    seqs_path: Path,
    model_path: Path,
    num_workers: int,
    embedding_size: int,
    rnn_num_layers: int,
    rnn_dropout_rate: float,
    variational_dropout: bool,
    bi_reduce: str,
    embedding_droprate: float,
    num_epochs: int,
    vocab_size: int,
    model_name: str,
    chkpt_dir: Path,
    log_path: Path,
    hyperparams_path: Path,
    trunc_len: int,
    seed: int,
):
    hyperparameters = {
        "architecture": "EncoderBarlow",
        "batch_size": batch_size,
        "dataset_path": str(dataset_path),
        "seqs_path": str(seqs_path),
        "model_path": str(model_path),
        "num_workers": num_workers,
        "embedding_size": embedding_size,
        "rnn_num_layers": rnn_num_layers,
        "rnn_dropout_rate": rnn_dropout_rate,
        "variational_dropout": variational_dropout,
        "bi_reduce": bi_reduce,
        "embedding_droprate": embedding_droprate,
        "num_epochs": num_epochs,
        "vocab_size": vocab_size,
        "model_name": model_name,
        "chkpt_dir": str(chkpt_dir),
        "log_path": str(log_path),
        "hyperparams_path": str(hyperparams_path),
        "trunc_len": trunc_len,
        "seed": seed,
    }

    makedirs(chkpt_dir.parent, exist_ok=True)
    makedirs(log_path.parent, exist_ok=True)
    makedirs(hyperparams_path.parent, exist_ok=True)

    with open(hyperparams_path, "w") as f:
        json.dump(hyperparameters, f, indent=3)

    seed_everything(seed)

    dict_logger = DictLogger()
    data_module = OmaTripletDataModule(
        batch_size, dataset_path, seqs_path, model_path, num_workers, trunc_len
    )
    data_module.setup("training")
    steps_per_epoch = len(data_module.train_dataloader())

    model = make_rnn_barlow_encoder(
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

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=chkpt_dir,
        filename=model_name + "-{epoch:02d}-{val_loss:.2f}",
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=num_epochs,
        precision=16,
        logger=[dict_logger],
        callbacks=[checkpoint_callback],
        deterministic=True,
    )
    trainer.fit(model, data_module)

    test_results = trainer.test(dataloaders=data_module, ckpt_path="best")

    dict_logger.metrics["test_results"] = test_results

    with open(log_path, "w") as f:
        json.dump(dict_logger.metrics, f, indent=3)


def train_transformers(
    batch_size: int,
    dataset_path: Path,
    seqs_path: Path,
    model_path: Path,
    num_workers: int,
    feedforward_size: int,
    embedding_size: int,
    num_layers: int,
    num_heads: int,
    dropout_rate: float,
    embedding_droprate: float,
    num_epochs: int,
    layer_norm: float,
    vocab_size: int,
    model_name: str,
    chkpt_dir: Path,
    log_path: Path,
    hyperparams_path: Path,
    trunc_len: int,
    seed: int,
):
    hyperparameters = {
        "architecture": "EncoderTransformerBarlow",
        "batch_size": batch_size,
        "dataset_path": str(dataset_path),
        "seqs_path": str(seqs_path),
        "model_path": str(model_path),
        "num_workers": num_workers,
        "feedforward_size": feedforward_size,
        "embedding_size": embedding_size,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "dropout_rate": dropout_rate,
        "embedding_droprate": embedding_droprate,
        "num_epochs": num_epochs,
        "layer_norm": layer_norm,
        "vocab_size": vocab_size,
        "model_name": model_name,
        "chkpt_dir": str(chkpt_dir),
        "log_path": str(log_path),
        "hyperparams_path": str(hyperparams_path),
        "trunc_len": trunc_len,
        "seed": seed,
    }

    makedirs(chkpt_dir.parent, exist_ok=True)
    makedirs(log_path.parent, exist_ok=True)
    makedirs(hyperparams_path.parent, exist_ok=True)

    with open(hyperparams_path, "w") as f:
        json.dump(hyperparameters, f, indent=3)

    seed_everything(seed)

    dict_logger = DictLogger()
    data_module = OmaTripletDataModule(
        batch_size, dataset_path, seqs_path, model_path, num_workers, trunc_len
    )
    data_module.setup("training")
    steps_per_epoch = len(data_module.train_dataloader())

    model = make_transformers_barlow_encoder(
        vocab_size=vocab_size,
        embedding_size=embedding_size,
        num_layers=num_layers,
        dropout_rate=dropout_rate,
        feedforward_size=feedforward_size,
        num_heads=num_heads,
        activation_fn="gelu",
        layer_norm=layer_norm,
        batch_size=batch_size,
        embedding_droprate=embedding_droprate,
        num_epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        trunc_len=trunc_len,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=chkpt_dir,
        filename=model_name + "-{epoch:02d}-{val_loss:.2f}",
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=num_epochs,
        precision=16,
        logger=[dict_logger],
        callbacks=[checkpoint_callback],
        deterministic=True,
    )
    trainer.fit(model, data_module)

    test_results = trainer.test(dataloaders=data_module, ckpt_path="best")

    dict_logger.metrics["test_results"] = test_results

    with open(log_path, "w") as f:
        json.dump(dict_logger.metrics, f, indent=3)
