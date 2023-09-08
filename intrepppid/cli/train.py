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

from passlib import pwd
from pathlib import Path
from datetime import datetime
from typing import Optional
from random import randint

from intrepppid.e2e.e2e_barlow import train_e2e_rnn_barlow
from intrepppid.e2e.e2e_transformer import train_e2e_transformer_barlow, train_e2e_transformer_barlow_attn_head
from intrepppid.e2e.e2e_triplet import train_e2e_rnn_triplet, train_e2e_transformer_triplet
from intrepppid.e2e.e2e_cos import train_e2e_rnn_cos
from intrepppid.e2e.e2e_vanilla import train_e2e_transformer_vanilla_attention
from intrepppid.encoders.barlow import train_rnn as barlow_train_rnn
from intrepppid.encoders.barlow import train_transformers as barlow_train_transformers
from intrepppid.classifier.barlow import (
    train_classifier_barlow,
    train_classifier_transformers_barlow,
)


class Train(object):
    @staticmethod
    def encoder_barlow(
        dataset_path: Path,
        seqs_path: Path,
        model_path: Path,
        num_epochs: int,
        batch_size: int = 20,
        num_workers: int = 4,
        embedding_size: int = 64,
        rnn_num_layers: int = 2,
        rnn_dropout_rate: float = 0.3,
        variational_dropout: bool = False,
        bi_reduce: str = "last",
        embedding_droprate: float = 0.3,
        vocab_size: int = 250,
        save_dir: Path = Path("./logs/encoder_barlow"),
        trunc_len: int = 1500,
        seed: Optional[int] = None,
    ):
        dataset_path = Path(dataset_path)
        seqs_path = Path(seqs_path)
        model_path = Path(model_path)
        save_dir = Path(save_dir)

        seed = seed if seed is not None else randint(0, 9999999)

        dt = datetime.now()
        dt = dt.strftime("%y.%j-%H.%M")

        model_name = pwd.genphrase(length=2, sep="-")
        model_name = f"{dt}-{model_name}"

        chkpt_dir = save_dir / f"{model_name}/chkpt"
        log_path = save_dir / f"{model_name}/metrics.json"
        hyperparams_path = save_dir / f"{model_name}/hyperparams.json"

        barlow_train_rnn(
            batch_size,
            dataset_path,
            seqs_path,
            model_path,
            num_workers,
            embedding_size,
            rnn_num_layers,
            rnn_dropout_rate,
            variational_dropout,
            bi_reduce,
            embedding_droprate,
            num_epochs,
            vocab_size,
            model_name,
            chkpt_dir,
            log_path,
            hyperparams_path,
            trunc_len,
            seed,
        )

    @staticmethod
    def encoder_transformers_barlow(
        dataset_path: Path,
        seqs_path: Path,
        model_path: Path,
        num_epochs: int,
        batch_size: int = 20,
        num_workers: int = 4,
        embedding_size: int = 64,
        num_layers: int = 2,
        num_heads: int = 2,
        dropout_rate: float = 0.3,
        feedforward_size: int = 128,
        layer_norm: float = 1e-5,
        embedding_droprate: float = 0.3,
        vocab_size: int = 250,
        save_dir: Path = Path("./logs/encoder_transformers_barlow"),
        trunc_len: int = 1500,
        seed: Optional[int] = None,
    ):
        seed = seed if seed is not None else randint(0, 9999999)

        dt = datetime.now()
        dt = dt.strftime("%y.%j-%H.%M")

        model_name = pwd.genphrase(length=2, sep="-")
        model_name = f"{dt}-{model_name}"

        chkpt_dir = save_dir / model_name / "chkpt"
        log_path = save_dir / model_name / "metrics.json"
        hyperparams_path = save_dir / model_name / "hyperparams.json"

        barlow_train_transformers(
            batch_size,
            dataset_path,
            seqs_path,
            model_path,
            num_workers,
            feedforward_size,
            embedding_size,
            num_layers,
            num_heads,
            dropout_rate,
            embedding_droprate,
            num_epochs,
            layer_norm,
            vocab_size,
            model_name,
            chkpt_dir,
            log_path,
            hyperparams_path,
            trunc_len,
            seed,
        )

    @staticmethod
    def classifier_barlow(
        barlow_hyperparams_path: Path,
        barlow_checkpoint_path: Path,
        ppi_dataset_path: Path,
        num_epochs: int,
        batch_size: int,
        c_type: int,
        embedding_droprate: float = 0.3,
        do_rate: float = 0.3,
        workers: int = 4,
        seed: Optional[int] = None,
        fine_tune_epochs: Optional[int] = None,
        log_path: Path = Path("./logs/classifier_barlow"),
    ):
        dt = datetime.now()
        dt = dt.strftime("%y.%j-%H.%M")

        model_name = pwd.genphrase(length=2, sep="-")
        model_name = f"{dt}-{model_name}"

        chkpt_dir = log_path / model_name / "chkpt"
        hyperparams_path = log_path / model_name / "hyperparams.json"

        train_classifier_barlow(
            barlow_hyperparams_path,
            barlow_checkpoint_path,
            ppi_dataset_path,
            log_path,
            hyperparams_path,
            chkpt_dir,
            c_type,
            model_name,
            workers,
            embedding_droprate,
            do_rate,
            num_epochs,
            batch_size,
            seed,
            fine_tune_epochs,
        )

    @staticmethod
    def classifier_transformers_barlow(
        barlow_hyperparams_path: Path,
        barlow_checkpoint_path: Path,
        ppi_dataset_path: Path,
        log_path: Path,
        c_type: int,
        workers: int,
        embedding_droprate: float,
        do_rate: float,
        num_epochs: int,
        batch_size: int,
        seed: int,
        fine_tune_epochs: int,
    ):
        log_path = Path(log_path)

        dt = datetime.now()
        dt = dt.strftime("%y.%j-%H.%M")

        model_name = pwd.genphrase(length=2, sep="-")
        model_name = f"{dt}-{model_name}"

        chkpt_dir = log_path / f"{model_name}/chkpt"
        hyperparams_path = log_path / f"{model_name}/hyperparams.json"

        train_classifier_transformers_barlow(
            barlow_hyperparams_path,
            barlow_checkpoint_path,
            ppi_dataset_path,
            log_path,
            hyperparams_path,
            chkpt_dir,
            c_type,
            model_name,
            workers,
            embedding_droprate,
            do_rate,
            num_epochs,
            batch_size,
            seed,
            fine_tune_epochs,
        )

    @staticmethod
    def e2e_rnn_barlow(
        ppi_dataset_path: Path,
        sentencepiece_path: Path,
        c_type: int,
        num_epochs: int,
        batch_size: int,
        seed: Optional[int] = None,
        vocab_size: int = 250,
        trunc_len: int = 1500,
        embedding_size: int = 64,
        rnn_num_layers: int = 2,
        rnn_dropout_rate: float = 0.3,
        variational_dropout: bool = False,
        bi_reduce: str = "last",
        workers: int = 4,
        embedding_droprate: float = 0.3,
        do_rate: float = 0.3,
        log_path: Path = Path("./logs/e2e_rnn_barlow"),
        encoder_only_steps: int = -1,
        classifier_warm_up: int = -1,
        beta_classifier: float = 4.0,
        use_projection: bool = True,
        projection_dropconnect: float = 0.3,
        optimizer_type: str = "ranger21",
        lr: float = 1e-2,
    ):
        dt = datetime.now()
        dt = dt.strftime("%y.%j-%H.%M")

        model_name = pwd.genphrase(length=2, sep="-")
        model_name = f"{dt}-{model_name}"

        log_path = Path(log_path)

        chkpt_dir = log_path / model_name / "chkpt"
        hyperparams_path = log_path / model_name / "hyperparams.json"

        train_e2e_rnn_barlow(
            vocab_size,
            trunc_len,
            embedding_size,
            rnn_num_layers,
            rnn_dropout_rate,
            variational_dropout,
            bi_reduce,
            ppi_dataset_path,
            sentencepiece_path,
            log_path,
            hyperparams_path,
            chkpt_dir,
            c_type,
            model_name,
            workers,
            embedding_droprate,
            do_rate,
            num_epochs,
            batch_size,
            encoder_only_steps,
            classifier_warm_up,
            beta_classifier,
            use_projection,
            projection_dropconnect,
            optimizer_type,
            lr,
            seed,
        )

    @staticmethod
    def e2e_transformer_barlow(
        ppi_dataset_path: Path,
        sentencepiece_path: Path,
        log_path: Path,
        c_type: int,
        num_epochs: int,
        batch_size: int,
        embedding_droprate: float = 0.3,
        do_rate: float = 0.3,
        workers: int = 4,
        encoder_only_steps: int = -1,
        classifier_warm_up: int = -1,
        beta_classifier: float = 1.0,
        use_projection: bool = True,
        projection_dropconnect: float = 0.3,
        optimizer_type: str = "ranger21",
        lr: float = 0.01,
        vocab_size: int = 250,
        trunc_len: int = 1500,
        embedding_size: int = 64,
        transformer_num_layers: int = 2,
        transformer_feedforward_size: int = 128,
        transformer_num_heads: int = 2,
        variational_dropout: bool = False,
        resume_checkpoint_path: Optional[Path] = None,
        fine_tune_mode: bool = False,
        seed: Optional[int] = None,
    ):
        dt = datetime.now()
        dt = dt.strftime("%y.%j-%H.%M")

        model_name = pwd.genphrase(length=2, sep="-")
        model_name = f"{dt}-{model_name}"

        log_path = Path(log_path)

        chkpt_dir = log_path / model_name / "chkpt"
        hyperparams_path = log_path / model_name / "hyperparams.json"

        train_e2e_transformer_barlow(
            vocab_size,
            trunc_len,
            embedding_size,
            transformer_num_layers,
            transformer_feedforward_size,
            transformer_num_heads,
            variational_dropout,
            ppi_dataset_path,
            sentencepiece_path,
            log_path,
            hyperparams_path,
            chkpt_dir,
            c_type,
            model_name,
            workers,
            embedding_droprate,
            do_rate,
            num_epochs,
            batch_size,
            encoder_only_steps,
            classifier_warm_up,
            beta_classifier,
            use_projection,
            projection_dropconnect,
            optimizer_type,
            lr,
            resume_checkpoint_path,
            fine_tune_mode,
            seed,
        )

    @staticmethod
    def e2e_rnn_triplet(
            ppi_dataset_path: Path,
            sentencepiece_path: Path,
            c_type: int,
            num_epochs: int,
            batch_size: int,
            seed: Optional[int] = None,
            vocab_size: int = 250,
            trunc_len: int = 1500,
            embedding_size: int = 64,
            rnn_num_layers: int = 2,
            rnn_dropout_rate: float = 0.3,
            variational_dropout: bool = False,
            bi_reduce: str = "last",
            workers: int = 4,
            embedding_droprate: float = 0.3,
            do_rate: float = 0.3,
            log_path: Path = Path("./logs/e2e_rnn_triplet"),
            encoder_only_steps: int = -1,
            classifier_warm_up: int = -1,
            beta_classifier: float = 4.0,
            use_projection: bool = False,
            checkpoint_path: Optional[Path] = None,
    ):
        dt = datetime.now()
        dt = dt.strftime("%y.%j-%H.%M")

        model_name = pwd.genphrase(length=2, sep="-")
        model_name = f"{dt}-{model_name}"

        log_path = Path(log_path)

        chkpt_dir = log_path / model_name / "chkpt"
        hyperparams_path = log_path / model_name / "hyperparams.json"

        train_e2e_rnn_triplet(
            vocab_size,
            trunc_len,
            embedding_size,
            rnn_num_layers,
            rnn_dropout_rate,
            variational_dropout,
            bi_reduce,
            ppi_dataset_path,
            sentencepiece_path,
            log_path,
            hyperparams_path,
            chkpt_dir,
            c_type,
            model_name,
            workers,
            embedding_droprate,
            do_rate,
            num_epochs,
            batch_size,
            encoder_only_steps,
            classifier_warm_up,
            beta_classifier,
            checkpoint_path,
            use_projection,
            seed,
        )

    @staticmethod
    def e2e_rnn_cos(
            ppi_dataset_path: Path,
            sentencepiece_path: Path,
            c_type: int,
            num_epochs: int,
            batch_size: int,
            seed: Optional[int] = None,
            vocab_size: int = 250,
            trunc_len: int = 1500,
            embedding_size: int = 64,
            rnn_num_layers: int = 2,
            rnn_dropout_rate: float = 0.3,
            variational_dropout: bool = False,
            bi_reduce: str = "last",
            workers: int = 4,
            embedding_droprate: float = 0.3,
            do_rate: float = 0.3,
            log_path: Path = Path("./logs/e2e_rnn_cos"),
            encoder_only_steps: int = -1,
            classifier_warm_up: int = -1,
            beta_classifier: float = 4.0,
            use_projection: bool = False,
            checkpoint_path: Optional[Path] = None,
            optimizer: str = "ranger21",
            lr: float = 1e-3
    ):
        dt = datetime.now()
        dt = dt.strftime("%y.%j-%H.%M")

        model_name = pwd.genphrase(length=2, sep="-")
        model_name = f"{dt}-{model_name}"

        log_path = Path(log_path)

        chkpt_dir = log_path / model_name / "chkpt"
        hyperparams_path = log_path / model_name / "hyperparams.json"

        train_e2e_rnn_cos(
            vocab_size,
            trunc_len,
            embedding_size,
            rnn_num_layers,
            rnn_dropout_rate,
            variational_dropout,
            bi_reduce,
            ppi_dataset_path,
            sentencepiece_path,
            log_path,
            hyperparams_path,
            chkpt_dir,
            c_type,
            model_name,
            workers,
            embedding_droprate,
            do_rate,
            num_epochs,
            batch_size,
            encoder_only_steps,
            classifier_warm_up,
            beta_classifier,
            checkpoint_path,
            use_projection,
            seed,
            optimizer,
            lr
        )

    @staticmethod
    def e2e_transformer_triplet(
            ppi_dataset_path: Path,
            sentencepiece_path: Path,
            c_type: int,
            num_epochs: int,
            batch_size: int,
            log_path: Path = Path("logs/e2e_transformer_triplet"),
            embedding_droprate: float = 0.3,
            do_rate: float = 0.3,
            workers: int = 4,
            encoder_only_steps: int = -1,
            classifier_warm_up: int = -1,
            beta_classifier: float = 1.0,
            use_projection: bool = True,
            projection_dropconnect: float = 0.3,
            optimizer_type: str = "ranger21",
            lr: float = 0.01,
            vocab_size: int = 250,
            trunc_len: int = 1500,
            embedding_size: int = 64,
            transformer_num_layers: int = 2,
            transformer_feedforward_size: int = 128,
            transformer_num_heads: int = 2,
            variational_dropout: bool = False,
            resume_checkpoint_path: Optional[Path] = None,
            fine_tune_mode: bool = False,
            seed: Optional[int] = None,
    ):
        dt = datetime.now()
        dt = dt.strftime("%y.%j-%H.%M")

        model_name = pwd.genphrase(length=2, sep="-")
        model_name = f"{dt}-{model_name}"

        log_path = Path(log_path)

        chkpt_dir = log_path / model_name / "chkpt"
        hyperparams_path = log_path / model_name / "hyperparams.json"

        train_e2e_transformer_triplet(
            vocab_size,
            trunc_len,
            embedding_size,
            transformer_num_layers,
            transformer_feedforward_size,
            transformer_num_heads,
            variational_dropout,
            ppi_dataset_path,
            sentencepiece_path,
            log_path,
            hyperparams_path,
            chkpt_dir,
            c_type,
            model_name,
            workers,
            embedding_droprate,
            do_rate,
            num_epochs,
            batch_size,
            encoder_only_steps,
            classifier_warm_up,
            beta_classifier,
            use_projection,
            projection_dropconnect,
            optimizer_type,
            lr,
            resume_checkpoint_path,
            fine_tune_mode,
            seed,
        )

    @staticmethod
    def e2e_transformer_triplet(
            ppi_dataset_path: Path,
            sentencepiece_path: Path,
            c_type: int,
            num_epochs: int,
            batch_size: int,
            log_path: Path = Path("logs/e2e_transformer_triplet"),
            embedding_droprate: float = 0.3,
            do_rate: float = 0.3,
            workers: int = 4,
            encoder_only_steps: int = -1,
            classifier_warm_up: int = -1,
            beta_classifier: float = 1.0,
            use_projection: bool = True,
            projection_dropconnect: float = 0.3,
            optimizer_type: str = "ranger21",
            lr: float = 0.01,
            vocab_size: int = 250,
            trunc_len: int = 1500,
            embedding_size: int = 64,
            transformer_num_layers: int = 2,
            transformer_feedforward_size: int = 128,
            transformer_num_heads: int = 2,
            variational_dropout: bool = False,
            resume_checkpoint_path: Optional[Path] = None,
            fine_tune_mode: bool = False,
            seed: Optional[int] = None,
    ):
        dt = datetime.now()
        dt = dt.strftime("%y.%j-%H.%M")

        model_name = pwd.genphrase(length=2, sep="-")
        model_name = f"{dt}-{model_name}"

        log_path = Path(log_path)

        chkpt_dir = log_path / model_name / "chkpt"
        hyperparams_path = log_path / model_name / "hyperparams.json"

        train_e2e_transformer_triplet(
            vocab_size,
            trunc_len,
            embedding_size,
            transformer_num_layers,
            transformer_feedforward_size,
            transformer_num_heads,
            variational_dropout,
            ppi_dataset_path,
            sentencepiece_path,
            log_path,
            hyperparams_path,
            chkpt_dir,
            c_type,
            model_name,
            workers,
            embedding_droprate,
            do_rate,
            num_epochs,
            batch_size,
            encoder_only_steps,
            classifier_warm_up,
            beta_classifier,
            use_projection,
            projection_dropconnect,
            optimizer_type,
            lr,
            resume_checkpoint_path,
            fine_tune_mode,
            seed,
        )


    @staticmethod
    def e2e_transformer_barlow_attention_head(
            ppi_dataset_path: Path,
            sentencepiece_path: Path,
            c_type: int,
            num_epochs: int,
            batch_size: int,
            log_path: Path = Path("logs/e2e_transformer_triplet"),
            embedding_droprate: float = 0.3,
            do_rate: float = 0.3,
            workers: int = 4,
            encoder_only_steps: int = -1,
            classifier_warm_up: int = -1,
            beta_classifier: float = 1.0,
            use_projection: bool = False,
            projection_dropconnect: float = 0.3,
            optimizer_type: str = "ranger21",
            lr: float = 0.01,
            vocab_size: int = 250,
            trunc_len: int = 1500,
            embedding_size: int = 64,
            transformer_num_layers: int = 2,
            transformer_feedforward_size: int = 128,
            transformer_num_heads: int = 2,
            variational_dropout: bool = False,
            resume_checkpoint_path: Optional[Path] = None,
            fine_tune_mode: bool = False,
            seed: Optional[int] = None,
    ):
        dt = datetime.now()
        dt = dt.strftime("%y.%j-%H.%M")

        model_name = pwd.genphrase(length=2, sep="-")
        model_name = f"{dt}-{model_name}"

        log_path = Path(log_path)

        chkpt_dir = log_path / model_name / "chkpt"
        hyperparams_path = log_path / model_name / "hyperparams.json"

        train_e2e_transformer_barlow_attn_head(
            vocab_size,
            trunc_len,
            embedding_size,
            transformer_num_layers,
            transformer_feedforward_size,
            transformer_num_heads,
            variational_dropout,
            ppi_dataset_path,
            sentencepiece_path,
            log_path,
            hyperparams_path,
            chkpt_dir,
            c_type,
            model_name,
            workers,
            embedding_droprate,
            do_rate,
            num_epochs,
            batch_size,
            encoder_only_steps,
            classifier_warm_up,
            beta_classifier,
            use_projection,
            projection_dropconnect,
            optimizer_type,
            lr,
            resume_checkpoint_path,
            fine_tune_mode,
            seed,
        )

    @staticmethod
    def e2e_transformer_vanilla_attention_head(
            ppi_dataset_path: Path,
            sentencepiece_path: Path,
            c_type: int,
            num_epochs: int,
            batch_size: int,
            log_path: Path = Path("logs/e2e_transformer_vanilla_attention_head"),
            embedding_droprate: float = 0.3,
            do_rate: float = 0.3,
            workers: int = 4,
            encoder_only_steps: int = -1,
            optimizer_type: str = "ranger21",
            lr: float = 0.01,
            vocab_size: int = 250,
            trunc_len: int = 1500,
            embedding_size: int = 64,
            transformer_num_layers: int = 2,
            transformer_feedforward_size: int = 128,
            transformer_num_heads: int = 2,
            cross_attention_heads: int = 1,
            gpu: int = 0,
            seed: Optional[int] = None,
    ):
        dt = datetime.now()
        dt = dt.strftime("%y.%j-%H.%M")

        model_name = pwd.genphrase(length=2, sep="-")
        model_name = f"{dt}-{model_name}"

        log_path = Path(log_path)

        chkpt_dir = log_path / model_name / "chkpt"
        hyperparams_path = log_path / model_name / "hyperparams.json"

        train_e2e_transformer_vanilla_attention(
            vocab_size,
            trunc_len,
            embedding_size,
            transformer_num_layers,
            transformer_feedforward_size,
            transformer_num_heads,
            ppi_dataset_path,
            sentencepiece_path,
            log_path,
            hyperparams_path,
            chkpt_dir,
            c_type,
            model_name,
            workers,
            embedding_droprate,
            do_rate,
            num_epochs,
            batch_size,
            encoder_only_steps,
            optimizer_type,
            lr,
            cross_attention_heads,
            gpu,
            seed,
        )
