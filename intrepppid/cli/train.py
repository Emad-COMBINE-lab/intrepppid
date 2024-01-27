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
from typing import Optional, Union

from intrepppid.e2e.e2e_triplet import train_e2e_rnn_triplet


class Train(object):

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
            lr: Union[float, str] = 1e-2,
            use_projection: bool = False,
            checkpoint_path: Optional[Path] = None,
            optimizer_type: str = "ranger21"
    ):
        """
        Train INTREPPPID in an end-to-end fashion using an AWD-LSTM encoder and MLP classifier.

        :param ppi_dataset_path: Path to the PPI dataset. Must be in the INTREPPPID HDF5 format.
        :param sentencepiece_path: Path to the SentencePiece model.
        :param c_type: Specifies which dataset in the INTREPPPID HDF5 dataset to use by specifying the C-type.
        :param num_epochs: Number of epochs to train the model for.
        :param batch_size: The number of samples to use in the batch.
        :param seed: The random seed. If not specified, chosen at random.
        :param vocab_size: The number of tokens in the SentencePiece vocabulary. Defaults to 250.
        :param trunc_len: Length at which to truncate sequences. Defaults to 1500.
        :param embedding_size: The size of embeddings. Defaults to 64.
        :param rnn_num_layers: The number of layers in the AWD-LSTM encoder to use. Defaults to 2.
        :param rnn_dropout_rate: The dropconnect rate for the AWD-LSTM encoder. Defaults to 0.3.
        :param variational_dropout: Whether to use variational dropout, as described in the AWD-LSTM manuscript. Defaults to True.
        :param bi_reduce: Method to reduce the two LSTM embeddings for both directions. Must be one of "concat", "max", "mean", "last". Defaults to "last".
        :param workers: The number of processes to use for the DataLoader. Defaults to 4.
        :param embedding_droprate: The amount of Embedding Dropout to use (a la AWD-LSTM). Defaults to 0.3.
        :param do_rate: The amount of dropout to use in the MLP Classifier. Defaults to 0.3.
        :param log_path: The path to save logs. Defaults to "./logs/e2e_rnn_triplet".
        :param encoder_only_steps: The number of steps to only train the encoder and not the classifier. Defaults to -1 (No steps).
        :param classifier_warm_up: The number of steps to only train the classifier and not the encoder. Defaults to -1 (No steps).
        :param beta_classifier: Adjusts the amount of weight to give the PPI Classification loss, relative to the Orthologue Locality loss. The loss becomes (1/beta_classifier)*classifier loss + [1-(1/beta_classifier)]*orthologue_loss. Defaults to 1 (equal contribution of both losses).
        :param lr: Learning rate to use. Defaults to 1e-2.
        :param use_projection: Whether to use a projection network after the encoder. Defaults to False.
        :param checkpoint_path: The location where checkpoints are to be saved. Defaults to log_path / model_name / "chkpt"
        :param optimizer_type: The optimizer to use while training. Must be one of "ranger21", "ranger21_xx", "adamw", "adamw_1cycle", or "adamw_cosine". Defaults to "ranger21".
        """
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
            lr,
            checkpoint_path,
            use_projection,
            optimizer_type,
            seed,
        )
