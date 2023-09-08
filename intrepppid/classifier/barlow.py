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

from intrepppid.encoders import (
    make_rnn_barlow_encoder,
    make_transformers_barlow_encoder,
)
from intrepppid.data import RapppidDataModule2
from intrepppid.classifier.classifynet import ClassifyNet
from intrepppid.utils import DictLogger
from intrepppid.classifier.head import MLPHead
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
import torch
from pathlib import Path
from typing import Optional
from os import makedirs
import json


def make_classifier_barlow(
    barlow_hyperparams_path: Path,
    barlow_checkpoint_path: Path,
    embedding_droprate: float,
    do_rate: float,
    steps_per_epoch: int,
    num_epochs: int,
    fine_tune_epochs: int,
):
    with open(barlow_hyperparams_path) as f:
        barlow_hyperparams = json.load(f)

    encoder = make_rnn_barlow_encoder(
        barlow_hyperparams["vocab_size"],
        barlow_hyperparams["embedding_size"],
        barlow_hyperparams["rnn_num_layers"],
        barlow_hyperparams["rnn_dropout_rate"],
        barlow_hyperparams["variational_dropout"],
        barlow_hyperparams["bi_reduce"],
        barlow_hyperparams["batch_size"],
        barlow_hyperparams["embedding_droprate"],
        barlow_hyperparams["num_epochs"],
        steps_per_epoch,
    )

    weights = torch.load(barlow_checkpoint_path)["state_dict"]

    encoder.load_state_dict(weights)
    encoder.eval()

    head = MLPHead(barlow_hyperparams["embedding_size"], do_rate)

    net = ClassifyNet(
        encoder, head, embedding_droprate, num_epochs, steps_per_epoch, fine_tune_epochs
    )

    return net


def make_classifier_transformers_barlow(
    barlow_hyperparams_path: Path,
    barlow_checkpoint_path: Path,
    embedding_droprate: float,
    do_rate: float,
    steps_per_epoch: int,
    num_epochs: int,
    fine_tune_epochs: int,
):
    with open(barlow_hyperparams_path) as f:
        barlow_hyperparams = json.load(f)

    encoder = make_transformers_barlow_encoder(
        barlow_hyperparams["vocab_size"],
        barlow_hyperparams["embedding_size"],
        barlow_hyperparams["num_layers"],
        barlow_hyperparams["dropout_rate"],
        barlow_hyperparams["feedforward_size"],
        barlow_hyperparams["num_heads"],
        "gelu",
        barlow_hyperparams["layer_norm"],
        barlow_hyperparams["batch_size"],
        barlow_hyperparams["embedding_droprate"],
        barlow_hyperparams["num_epochs"],
        steps_per_epoch,
        barlow_hyperparams["trunc_len"],
    )

    weights = torch.load(barlow_checkpoint_path)["state_dict"]

    encoder.load_state_dict(weights)
    encoder.eval()

    head = MLPHead(barlow_hyperparams["embedding_size"], do_rate)

    net = ClassifyNet(
        encoder, head, embedding_droprate, num_epochs, steps_per_epoch, fine_tune_epochs
    )

    return net


def train_classifier_barlow(
    barlow_hyperparams_path: Path,
    barlow_checkpoint_path: Path,
    ppi_dataset_path: Path,
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
    seed: Optional[int] = None,
    fine_tune_epochs: Optional[int] = None,
):
    makedirs(chkpt_dir, exist_ok=True)
    makedirs(log_path, exist_ok=True)
    makedirs(hyperparams_path.parent, exist_ok=True)

    with open(barlow_hyperparams_path) as f:
        barlow_hyperparams = json.load(f)

    seed = barlow_hyperparams["seed"] if seed is None else seed

    seed_everything(seed)

    hyperparameters = {
        "architecture": "ClassifierBarlow",
        "barlow_hyperparams_path": str(barlow_hyperparams_path),
        "barlow_checkpoint_path": str(barlow_checkpoint_path),
        "ppi_dataset_path": str(ppi_dataset_path),
        "log_path": str(log_path),
        "hyperparams_path": str(hyperparams_path),
        "chkpt_dir": str(chkpt_dir),
        "model_name": model_name,
        "workers": workers,
        "embedding_droprate": embedding_droprate,
        "do_rate": do_rate,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "seed": seed,
        "fine_tune_epochs": fine_tune_epochs,
    }

    with open(hyperparams_path, "w") as f:
        json.dump(hyperparameters, f)

    data_module = RapppidDataModule2(
        batch_size=batch_size,
        dataset_path=ppi_dataset_path,
        c_type=c_type,
        trunc_len=barlow_hyperparams["trunc_len"],
        workers=workers,
        vocab_size=barlow_hyperparams["vocab_size"],
        model_file=barlow_hyperparams["model_path"],
        seed=seed,
    )

    data_module.setup("training")
    steps_per_epoch = len(data_module.train_dataloader())

    net = make_classifier_barlow(
        barlow_hyperparams["hyperparams_path"],
        barlow_checkpoint_path,
        embedding_droprate,
        do_rate,
        steps_per_epoch,
        num_epochs,
        fine_tune_epochs,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=chkpt_dir,
        filename=model_name + "-{epoch:02d}-{val_loss:.2f}",
    )

    dict_logger = DictLogger()

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=num_epochs,
        precision=16,
        logger=[dict_logger],
        callbacks=[checkpoint_callback],
    )

    trainer.fit(net, data_module)

    test_results = trainer.test(dataloaders=data_module, ckpt_path="best")

    dict_logger.metrics["test_results"] = test_results

    with open(log_path / model_name / "metrics.json", "w") as f:
        json.dump(dict_logger.metrics, f, indent=3)


def train_classifier_transformers_barlow(
    barlow_hyperparams_path: Path,
    barlow_checkpoint_path: Path,
    ppi_dataset_path: Path,
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
    seed: Optional[int] = None,
    fine_tune_epochs: Optional[int] = None,
):
    makedirs(chkpt_dir, exist_ok=True)
    makedirs(log_path, exist_ok=True)
    makedirs(hyperparams_path.parent, exist_ok=True)

    with open(barlow_hyperparams_path) as f:
        barlow_hyperparams = json.load(f)

    seed = barlow_hyperparams["seed"] if seed is None else seed

    seed_everything(seed)

    hyperparameters = {
        "architecture": "ClassifierTransformersBarlow",
        "barlow_hyperparams_path": str(barlow_hyperparams_path),
        "barlow_checkpoint_path": str(barlow_checkpoint_path),
        "ppi_dataset_path": str(ppi_dataset_path),
        "log_path": str(log_path),
        "hyperparams_path": str(hyperparams_path),
        "chkpt_dir": str(chkpt_dir),
        "model_name": model_name,
        "workers": workers,
        "embedding_droprate": embedding_droprate,
        "do_rate": do_rate,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "seed": seed,
        "fine_tune_epochs": fine_tune_epochs,
    }

    with open(hyperparams_path, "w") as f:
        json.dump(hyperparameters, f)

    data_module = RapppidDataModule2(
        batch_size=batch_size,
        dataset_path=ppi_dataset_path,
        c_type=c_type,
        trunc_len=barlow_hyperparams["trunc_len"],
        workers=workers,
        vocab_size=barlow_hyperparams["vocab_size"],
        model_file=barlow_hyperparams["model_path"],
        seed=seed,
    )

    data_module.setup("training")
    steps_per_epoch = len(data_module.train_dataloader())

    net = make_classifier_transformers_barlow(
        barlow_hyperparams_path,
        barlow_checkpoint_path,
        embedding_droprate,
        do_rate,
        steps_per_epoch,
        num_epochs,
        fine_tune_epochs,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=chkpt_dir,
        filename=model_name + "-{epoch:02d}-{val_loss:.2f}",
    )

    dict_logger = DictLogger()

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=num_epochs,
        precision=16,
        logger=[dict_logger],
        callbacks=[checkpoint_callback],
    )

    trainer.fit(net, data_module)

    test_results = trainer.test(dataloaders=data_module, ckpt_path="best")

    dict_logger.metrics["test_results"] = test_results

    with open(log_path / model_name / "metrics.json", "w") as f:
        json.dump(dict_logger.metrics, f, indent=3)
