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
from torch import nn

from intrepppid.encoders.awd_lstm import AWDLSTMEncoder
from intrepppid.e2e.e2e_triplet import TripletE2ENet
from intrepppid.classifier.head import MLPHead


def intrepppid_network(
        steps_per_epoch: int,
        vocab_size: int = 250,
        embedding_size: int = 64,
        rnn_num_layers: int = 2,
        rnn_dropout_rate: float = 0.3,
        variational_dropout: bool = False,
        bi_reduce: str = "last",
        embedding_droprate: float = 0.3,
        num_epochs: int = 100,
        do_rate: float = 0.3,
        beta_classifier: int = 2,
        lr: float = 1e-2,
        use_projection: bool = False,
        optimizer_type: str = "ranger21_xx"
):
    """
    This builds a PyTorch nn.Module which represents the INTREPPPID network as
    defined in the manuscript.

    It assembles a TripletE2ENet with an AWD-LSTM encoder and an MLP classifier.

    :param steps_per_epoch: Number of mini-batch steps iterated over each epoch. Only really maters for training.
    :param vocab_size: The number of tokens in the SentencePiece vocabulary. Defaults to 250.
    :param embedding_size: The size of embeddings. Defaults to 64.
    :param rnn_num_layers: The number of layers in the AWD-LSTM encoder to use. Defaults to 2.
    :param rnn_dropout_rate: The dropconnect rate for the AWD-LSTM encoder. Defaults to 0.3.
    :param variational_dropout: Whether to use variational dropout, as described in the AWD-LSTM manuscript. Defaults to False.
    :param bi_reduce: Method to reduce the two LSTM embeddings for both directions. Must be one of "concat", "max", "mean", "last". Defaults to "last".
    :param embedding_droprate: The amount of Embedding Dropout to use (a la AWD-LSTM). Defaults to 0.3.
    :param num_epochs: Number of epochs to train the model for.
    :param do_rate: The amount of dropout to use in the MLP Classifier. Defaults to 0.3.
    :param beta_classifier: Adjusts the amount of weight to give the PPI Classification loss, relative to the Orthologue Locality loss. The loss becomes (1/beta_classifier)*classifier loss + [1-(1/beta_classifier)]*orthologue_loss. Defaults to 1 (equal contribution of both losses).
    :param lr: Learning rate to use. Defaults to 1e-2.
    :param use_projection: Whether to use a projection network after the encoder. Defaults to False.
    :param optimizer_type: The optimizer to use while training. Must be one of "ranger21", "ranger21_xx", "adamw", "adamw_1cycle", or "adamw_cosine". Defaults to "ranger21".
    """

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

    return net
