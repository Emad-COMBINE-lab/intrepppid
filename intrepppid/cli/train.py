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
from intrepppid.encoders.barlow import train as barlow_train


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
    ):
        dt = datetime.now()
        dt = dt.strftime("%y.%j-%H.%M")

        model_name = pwd.genphrase(length=2, sep="-")
        model_name = f"{dt}-{model_name}"

        chkpt_dir = save_dir / model_name / "chkpt"
        log_path = save_dir / model_name / "metrics.json"
        hyperparams_path = save_dir / model_name / "hyperparams.json"

        barlow_train(
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
        )
