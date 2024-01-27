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
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    StochasticWeightAveraging,
)
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from torch import nn
import torchmetrics
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts

from intrepppid.classifier.head import MLPHead
from intrepppid.utils import embedding_dropout, DictLogger
from torch.optim import AdamW
from ranger21 import Ranger21
from typing import Optional, Union
from intrepppid.encoders.awd_lstm import AWDLSTMEncoder
from intrepppid.data.ppi_oma import IntrepppidDataModule


class TripletE2ENet(pl.LightningModule):
    def __init__(
        self,
        embedding_size: int,
        encoder: nn.Module,
        head: nn.Module,
        embedding_droprate: float,
        num_epochs: int,
        steps_per_epoch: int,
        beta_classifier: float,
        use_projection: bool,
        optimizer_type: str,
        lr: float
    ):
        """
        Create an end-to-end INTREPPPID network which uses a triplet loss for the orthologue task.

        This is a scaffold which requires an encoder and classifier to be specified.

        :param embedding_size: The size of the embeddings to use.
        :param encoder: The encoder neural network to use to embed the amino acid sequences.
        :param head: The classifier neural network to use to classify the embedded sequences as interactors or not.
        :param embedding_droprate: The rate at which embeddings are dropped out.
        :param num_epochs: The number of epochs to train for.
        :param steps_per_epoch: Number of mini-batch steps iterated over each epoch. Only really maters for training.
        :param beta_classifier: Adjusts the amount of weight to give the PPI Classification loss, relative to the Orthologue Locality loss. The loss becomes (1/beta_classifier)*classifier loss + [1-(1/beta_classifier)]*orthologue_loss. Defaults to 1 (equal contribution of both losses).
        :param use_projection: Whether to use a projection network after the encoder.
        :param optimizer_type: The optimizer to use while training. Must be one of "ranger21", "ranger21_xx", "adamw", "adamw_1cycle", or "adamw_cosine".
        :param lr: Learning rate to use.
        """
        super().__init__()
        self.encoder = encoder
        self.embedding_droprate = embedding_droprate
        self.classifier_criterion = nn.BCEWithLogitsLoss()
        self.num_epochs = num_epochs
        self.steps_per_epoch = steps_per_epoch

        self.triplet_criterion = nn.TripletMarginLoss(margin=1.0, p=2)

        if use_projection:
            self.triplet_projection = nn.Sequential(
                nn.Mish(), nn.Linear(embedding_size, embedding_size)
            )

        self.auroc = torchmetrics.AUROC(task="binary")
        self.average_precision = torchmetrics.AveragePrecision(task="binary")
        self.mcc = torchmetrics.MatthewsCorrCoef(task="binary", threshold=0.5)
        self.precision_metric = torchmetrics.Precision(task="binary")
        self.recall = torchmetrics.Recall(task="binary")

        self.do_rate = 0.3
        self.head = head
        self.beta_classifier = beta_classifier

        self.optimizer_type = optimizer_type
        self.lr = lr

        self.use_projection = use_projection

    def embedding_dropout(self, embed, words, p=0.2):
        return embedding_dropout(self.training, embed, words, p)

    def forward(self, x1, x2):
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)

        y_hat = self.head(z1, z2)

        return y_hat

    def step(self, batch, stage):
        p1_seq, p2_seq, omid_anchor_seq, omid_positive_seq, omid_negative_seq, y = batch

        if self.use_projection:
            z_omid_anchor = self.triplet_projection(self.encoder(omid_anchor_seq))
            z_omid_positive = self.triplet_projection(self.encoder(omid_positive_seq))
            z_omid_negative = self.triplet_projection(self.encoder(omid_negative_seq))
        else:
            z_omid_anchor = self.encoder(omid_anchor_seq)
            z_omid_positive = self.encoder(omid_positive_seq)
            z_omid_negative = self.encoder(omid_negative_seq)

        triplet_loss = self.triplet_criterion(z_omid_anchor, z_omid_positive, z_omid_negative)

        y_hat = self(p1_seq, p2_seq).squeeze(1)

        classifier_loss = self.classifier_criterion(y_hat, y.float())

        norm_beta_ssl = 1 / self.beta_classifier
        norm_beta_classifier = 1 - norm_beta_ssl

        loss = norm_beta_classifier * classifier_loss + norm_beta_ssl * triplet_loss

        self.log(
            f"{stage}_classifier_loss",
            classifier_loss.detach(),
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )
        self.log(
            f"{stage}_triplet_loss",
            triplet_loss.detach(),
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )
        self.log(f"{stage}_loss", loss, on_epoch=True, on_step=False, prog_bar=True)

        self.log(
            f"{stage}_classifier_loss_step",
            classifier_loss.detach(),
            on_epoch=False,
            on_step=True,
            prog_bar=False,
        )
        self.log(
            f"{stage}_triplet_loss_step",
            triplet_loss.detach(),
            on_epoch=False,
            on_step=True,
            prog_bar=False,
        )
        self.log(
            f"{stage}_loss_step", loss, on_epoch=False, on_step=True, prog_bar=False
        )

        auroc = self.auroc(y_hat, y)
        self.log(f"{stage}_auroc", auroc.detach(), on_epoch=True, on_step=False)

        ap = self.average_precision(y_hat, y)
        self.log(f"{stage}_ap", ap.detach(), on_epoch=True, on_step=False)

        mcc = self.mcc(y_hat, y)
        self.log(f"{stage}_mcc", mcc.detach(), on_epoch=True, on_step=False)

        pr = self.precision_metric(y_hat, y)
        self.log(f"{stage}_precision", pr.detach(), on_epoch=True, on_step=False)

        rec = self.recall(y_hat, y)
        self.log(f"{stage}_rec", rec.detach(), on_epoch=True, on_step=False)

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

            return optimizer

        elif self.optimizer_type == "ranger21_xx":
            optimizer = Ranger21(
                self.parameters(),
                use_warmup=True,
                warmdown_active=True,
                lr=self.lr,
                weight_decay=1e-2,
                num_batches_per_epoch=self.steps_per_epoch,
                num_epochs=self.num_epochs,
                warmdown_start_pct=0.72,
            )

            return optimizer

        elif self.optimizer_type == "adamw":
            optimizer = AdamW(self.parameters(), lr=self.lr)

            return optimizer

        elif self.optimizer_type == "adamw_1cycle":
            optimizer = AdamW(self.parameters(), lr=self.lr)
            scheduler = OneCycleLR(optimizer, self.lr, epochs=self.num_epochs, steps_per_epoch=self.steps_per_epoch)

            return [optimizer], [scheduler]

        elif self.optimizer_type == "adamw_cosine":
            optimizer = AdamW(self.parameters(), lr=self.lr)
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

            return [optimizer], [scheduler]

        else:
            raise ValueError(
                'Expected one of "ranger21", "adamw", "ranger21_xx", or "adamw_1cycle" as the optimizer type.'
            )


def train_e2e_rnn_triplet(
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
    lr: Union[float, str] = 1e-2,
    checkpoint_path: Optional[Path] = None,
    use_projection: bool = True,
    optimizer_type: str = "ranger21",
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
        "lr": lr,
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
        "checkpoint_path": checkpoint_path,
        "use_projection": use_projection,
        "seed": seed,
        "optimizer_type": optimizer_type
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
        eos=False,
        negative_omid=True
    )

    data_module.setup("training")
    steps_per_epoch = len(data_module.train_dataloader())

    embedder = nn.Embedding(vocab_size, embedding_size, padding_idx=0)

    encoder = AWDLSTMEncoder(
        embedder,
        embedding_size,
        embedding_droprate,
        rnn_num_layers,
        rnn_dropout_rate,
        variational_dropout,
        bi_reduce
    )

    head = MLPHead(embedding_size, do_rate)

    if lr == "auto":
        lr = 1e-2

    net = TripletE2ENet(
        embedding_size,
        encoder,
        head,
        embedding_droprate,
        num_epochs,
        steps_per_epoch,
        beta_classifier,
        use_projection,
        optimizer_type,
        lr
    )

    num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)

    print("######")
    print(f"NUM PARAMS:{num_params}")
    print("######")

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

    if lr == "auto":
        lr_finder = trainer.tuner.lr_find(net, datamodule=data_module)
        new_lr = lr_finder.suggestion()
        hyperparameters["new_lr"] = new_lr

        print(f"Found LR: {new_lr}")

        net.lr = new_lr

        # Redeclaring this so I can update the SWA calback with the new LR
        swa = StochasticWeightAveraging(swa_lrs=new_lr)

        trainer = pl.Trainer(
            accelerator="gpu",
            devices=1,
            max_epochs=num_epochs,
            precision=16,
            logger=[dict_logger, tb_logger],
            callbacks=[checkpoint_callback, lr_monitor, swa],
            log_every_n_steps=2,
        )

    trainer.fit(net, data_module, ckpt_path=checkpoint_path)

    test_results = trainer.test(dataloaders=data_module, ckpt_path="best")

    dict_logger.metrics["test_results"] = test_results

    with open(log_path / model_name / "metrics.json", "w") as f:
        json.dump(dict_logger.metrics, f, indent=3)
