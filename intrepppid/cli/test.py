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
import pytorch_lightning as pl
import json
from pathlib import Path
from typing import Optional
from intrepppid.classifier import make_classifier_barlow
from intrepppid.data.ppi import RapppidDataModule2


class Test(object):
    @staticmethod
    def classifier_barlow(
            dataset_path: Path,
            c_type: int,
            checkpoint_path: Path,
            hyperparams_path: Path,
            workers: Optional[int]
    ):
        with open(hyperparams_path) as f:
            hyperparams = json.load(f)

        with open(hyperparams['barlow_hyperparams_path']) as f:
            barlow_hyperparams = json.load(f)

        data_module = RapppidDataModule2(
            batch_size=hyperparams['batch_size'],
            dataset_path=dataset_path,
            c_type=c_type,
            trunc_len=barlow_hyperparams["trunc_len"],
            workers=workers if workers is not None else hyperparams['workers'],
            vocab_size=barlow_hyperparams["vocab_size"],
            model_file=barlow_hyperparams["model_path"],
            seed=hyperparams['seed'],
        )

        net = make_classifier_barlow(hyperparams['barlow_hyperparams_path'],
                                     hyperparams['barlow_checkpoint_path'],
                                     hyperparams['embedding_droprate'],
                                     hyperparams['do_rate'],
                                     1,
                                     1,
                                     1)

        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['state_dict'])
        net.eval()

        trainer = pl.Trainer(
            accelerator="gpu",
            devices=1,
            max_epochs=hyperparams['num_epochs'],
            precision=16
        )

        test_results = trainer.test(net, dataloaders=data_module)

        print(test_results)

