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

from intrepppid.classifier.head import MLPHead, AttentionHead
from intrepppid.utils import DictLogger
from typing import Optional, Tuple
from intrepppid.encoders.barlow import BarlowTwinsLoss, make_transformers_barlow_encoder
from intrepppid.data.ppi_oma import IntrepppidDataModule
from intrepppid.e2e.e2e_barlow import BarlowE2ENet


def train_e2e_transformer_barlow(
    vocab_size: int,
    trunc_len: int,
    embedding_size: int,
    transformer_num_layers: int,
    transformer_feedforward_size: int,
    transformer_num_heads: int,
    variational_dropout: bool,
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
    resume_checkpoint_path: Optional[Path] = None,
    fine_tune_mode: bool = False,
    seed: Optional[int] = None,
):
    makedirs(chkpt_dir, exist_ok=True)
    makedirs(log_path, exist_ok=True)
    makedirs(hyperparams_path.parent, exist_ok=True)

    seed = random.randint(0, 99999) if seed is None else seed

    seed_everything(seed)

    hyperparameters = {
        "architecture": "E2ETransformerBarlow",
        "vocab_size": vocab_size,
        "trunc_len": trunc_len,
        "embedding_size": embedding_size,
        "transformer_num_layers": transformer_num_layers,
        "transformer_feedforward_size": transformer_feedforward_size,
        "transformer_num_heads": transformer_num_heads,
        "variational_dropout": variational_dropout,
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
        "resume_checkpoint_path": resume_checkpoint_path,
        "fine_tune_mode": fine_tune_mode,
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
        sos=True,
        eos=True,
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
    )

    head = MLPHead(embedding_size, do_rate)

    net = BarlowE2ENet(
        embedding_size=embedding_size,
        batch_size=batch_size,
        encoder=encoder,
        head=head,
        embedding_droprate=embedding_droprate,
        num_epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        beta_classifier=beta_classifier,
        use_projection=use_projection,
        projection_dropconnect=projection_dropconnect,
        optimizer_type=optimizer_type,
        lr=lr,
    )

    # net = BarlowE2ENet(
    #    embedding_size, batch_size, encoder, head, embedding_droprate, num_epochs, steps_per_epoch, encoder_only_steps,
    #    classifier_warm_up, beta_classifier, use_projection, projection_dropconnect, optimizer_type, lr
    # )

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

    trainer.fit(net, data_module, ckpt_path=resume_checkpoint_path)

    test_results = trainer.test(dataloaders=data_module, ckpt_path="best")

    dict_logger.metrics["test_results"] = test_results

    with open(log_path / model_name / "metrics.json", "w") as f:
        json.dump(dict_logger.metrics, f, indent=3)


def train_e2e_transformer_barlow_attn_head(
    vocab_size: int,
    trunc_len: int,
    embedding_size: int,
    transformer_num_layers: int,
    transformer_feedforward_size: int,
    transformer_num_heads: int,
    variational_dropout: bool,
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
    resume_checkpoint_path: Optional[Path] = None,
    fine_tune_mode: bool = False,
    seed: Optional[int] = None,
):
    makedirs(chkpt_dir, exist_ok=True)
    makedirs(log_path, exist_ok=True)
    makedirs(hyperparams_path.parent, exist_ok=True)

    seed = random.randint(0, 99999) if seed is None else seed

    seed_everything(seed)

    hyperparameters = {
        "architecture": "E2ETransformerBarlow",
        "vocab_size": vocab_size,
        "trunc_len": trunc_len,
        "embedding_size": embedding_size,
        "transformer_num_layers": transformer_num_layers,
        "transformer_feedforward_size": transformer_feedforward_size,
        "transformer_num_heads": transformer_num_heads,
        "variational_dropout": variational_dropout,
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
        "resume_checkpoint_path": resume_checkpoint_path,
        "fine_tune_mode": fine_tune_mode,
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
        sos=True,
        eos=True,
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
        mean=True,
        truncate_to_longest=False
    )

    head = AttentionHead(embedding_size, do_rate, 1)

    net = BarlowE2ENet(
        embedding_size=embedding_size,
        batch_size=batch_size,
        encoder=encoder,
        head=head,
        embedding_droprate=embedding_droprate,
        num_epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        beta_classifier=beta_classifier,
        use_projection=use_projection,
        projection_dropconnect=projection_dropconnect,
        optimizer_type=optimizer_type,
        lr=lr,
    )

    # net = BarlowE2ENet(
    #    embedding_size, batch_size, encoder, head, embedding_droprate, num_epochs, steps_per_epoch, encoder_only_steps,
    #    classifier_warm_up, beta_classifier, use_projection, projection_dropconnect, optimizer_type, lr
    # )

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

    trainer.fit(net, data_module, ckpt_path=resume_checkpoint_path)

    test_results = trainer.test(dataloaders=data_module, ckpt_path="best")

    dict_logger.metrics["test_results"] = test_results

    with open(log_path / model_name / "metrics.json", "w") as f:
        json.dump(dict_logger.metrics, f, indent=3)
