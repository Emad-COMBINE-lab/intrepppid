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
import pytorch_lightning as pl
from pytorch_lightning.utilities.seed import seed_everything
import json
from pathlib import Path
from typing import Optional
from intrepppid.classifier import make_classifier_barlow
from intrepppid.encoders import make_rnn_barlow_encoder, make_transformers_barlow_encoder
from intrepppid.data.ppi import RapppidDataModule2
from intrepppid.data.ppi_oma import IntrepppidDataModule
from intrepppid.classifier.head import MLPHead
from intrepppid.e2e.e2e_barlow import BarlowE2ENet


class Test(object):
    @staticmethod
    def classifier_barlow(
        dataset_path: Path,
        c_type: int,
        checkpoint_path: Path,
        hyperparams_path: Path,
        workers: Optional[int],
        gpu: bool
    ):
        with open(hyperparams_path) as f:
            hyperparams = json.load(f)

        with open(hyperparams["barlow_hyperparams_path"]) as f:
            barlow_hyperparams = json.load(f)

        data_module = RapppidDataModule2(
            batch_size=hyperparams["batch_size"],
            dataset_path=dataset_path,
            c_type=c_type,
            trunc_len=barlow_hyperparams["trunc_len"],
            workers=workers if workers is not None else hyperparams["workers"],
            vocab_size=barlow_hyperparams["vocab_size"],
            model_file=barlow_hyperparams["model_path"],
            seed=hyperparams["seed"],
        )

        net = make_classifier_barlow(
            hyperparams["barlow_hyperparams_path"],
            hyperparams["barlow_checkpoint_path"],
            hyperparams["embedding_droprate"],
            hyperparams["do_rate"],
            1,
            1,
            1,
        )

        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint["state_dict"])
        net.eval()

        trainer = pl.Trainer(
            accelerator="gpu" if gpu else "cpu",
            devices=1,
            max_epochs=hyperparams["num_epochs"],
            precision=16,
        )

        test_results = trainer.test(net, dataloaders=data_module)

        print(test_results)

    @staticmethod
    def e2e_barlow(
        encoder_name: str,
        dataset_path: Path,
        c_type: int,
        checkpoint_path: Path,
        hyperparams_path: Path,
        workers: Optional[int],
        gpu: bool = False,
        head_name: str = "mlp",
        seed: Optional[int] = None
    ):
        with open(hyperparams_path) as f:
            hyperparams = json.load(f)

        seed = seed if seed is not None else hyperparams['seed']

        seed_everything(seed)

        data_module = IntrepppidDataModule(
            batch_size=hyperparams['batch_size'],
            dataset_path=dataset_path,
            c_type=c_type,
            trunc_len=hyperparams['trunc_len'],
            workers=workers,
            vocab_size=hyperparams['vocab_size'],
            model_file=hyperparams['sentencepiece_path'],
            seed=hyperparams['seed'],
            sos=True if encoder_name == "transformer" else False,
            eos=True if encoder_name == "transformer" else False,
        )

        if encoder_name == "rnn":
            encoder = make_rnn_barlow_encoder(
                hyperparams['vocab_size'],
                hyperparams['embedding_size'],
                hyperparams['rnn_num_layers'],
                hyperparams['rnn_dropout_rate'],
                hyperparams['variational_dropout'],
                hyperparams['bi_reduce'],
                hyperparams['batch_size'],
                hyperparams['embedding_droprate'],
                hyperparams['num_epochs'],
                0,
            )
        elif encoder_name == "transformer":
            encoder = make_transformers_barlow_encoder(
                hyperparams['vocab_size'],
                hyperparams['embedding_size'],
                hyperparams['transformer_num_layers'],
                hyperparams['do_rate'],
                hyperparams['transformer_feedforward_size'],
                hyperparams['transformer_num_heads'],
                nn.Mish(),
                1e-5,
                hyperparams['batch_size'],
                hyperparams['embedding_droprate'],
                hyperparams['num_epochs'],
                0,
                hyperparams['trunc_len'],
            )
        else:
            raise ValueError("Unexpected encoder name. Expected one of 'rnn' or 'transformer'.")

        if head_name == "mlp":
            head = MLPHead(hyperparams['embedding_size'], hyperparams['do_rate'])
        else:
            raise ValueError("Unexpected encoder name. Expected 'mlp'.")

        net = BarlowE2ENet(
            embedding_size=hyperparams['embedding_size'],
            batch_size=hyperparams['batch_size'],
            encoder=encoder,
            head=head,
            embedding_droprate=hyperparams['embedding_droprate'],
            num_epochs=hyperparams['num_epochs'],
            steps_per_epoch=0,
            beta_classifier=hyperparams['beta_classifier'],
            use_projection=hyperparams['use_projection'],
            projection_dropconnect=hyperparams['projection_dropconnect'],
            optimizer_type=hyperparams['optimizer_type'],
            lr=hyperparams['lr'],
        )

        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint["state_dict"])
        net.eval()

        trainer = pl.Trainer(
            accelerator="gpu" if gpu else "cpu",
            devices=1,
            max_epochs=hyperparams["num_epochs"],
            precision=16,
        )

        test_results = trainer.test(net, dataloaders=data_module)

        print(test_results)
