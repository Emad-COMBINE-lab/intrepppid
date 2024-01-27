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

import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import sentencepiece as sp
import numpy as np
from intrepppid.data.utils import encode_seq
import tables as tb
import torch
from pathlib import Path


class RapppidDataset2(Dataset):
    def __init__(self, dataset_path, c_type, split, model_file, trunc_len=1000):
        super().__init__()

        self.trunc_len = trunc_len
        self.dataset_path = dataset_path
        self.c_type = c_type
        self.split = split

        if self.split in ["test", "val"]:
            self.sampling = False
        else:
            self.sampling = True

        self.spp = sp.SentencePieceProcessor(model_file=model_file)

    @staticmethod
    def static_encode(
        trunc_len: int, spp, seq: str, sp: bool = True, pad: bool = True, sampling=True
    ):
        seq = seq[:trunc_len]

        if sp:
            toks = np.array(
                spp.encode(seq, enable_sampling=sampling, alpha=0.1, nbest_size=-1)
            )
        else:
            toks = encode_seq(seq)

        if pad:
            pad_len = trunc_len - len(toks)
            toks = np.pad(toks, (0, pad_len), "constant")

        return toks

    def encode(self, seq: str, sp: bool = True, pad: bool = True):
        return self.static_encode(self.trunc_len, self.spp, seq, sp, pad, self.sampling)

    def get_sequence(self, name: str):
        with tb.open_file(self.dataset_path) as dataset:
            seq = dataset.root.sequences.read_where(f'name=="{name}"')[0][1].decode(
                "utf8"
            )

        return seq

    def __getitem__(self, idx):
        with tb.open_file(self.dataset_path) as dataset:
            p1, p2, label = dataset.root["interactions"][f"c{self.c_type}"][
                f"c{self.c_type}_{self.split}"
            ][idx]

        p1 = p1.decode("utf8")
        p2 = p2.decode("utf8")

        p1_seq = self.encode(self.get_sequence(p1), sp=True, pad=True)

        p2_seq = self.encode(self.get_sequence(p2), sp=True, pad=True)

        p1_seq = torch.tensor(p1_seq).long()
        p2_seq = torch.tensor(p2_seq).long()
        label = torch.tensor(label).long()

        return p1_seq, p2_seq, label

    def __len__(self):
        with tb.open_file(self.dataset_path) as dataset:
            l = len(
                dataset.root["interactions"][f"c{self.c_type}"][
                    f"c{self.c_type}_{self.split}"
                ]
            )
        return l


class RapppidDataModule2(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        dataset_path: Path,
        c_type: int,
        trunc_len: int,
        workers: int,
        vocab_size: int,
        model_file: str,
        seed: int,
    ):
        super().__init__()

        sp.set_random_generator_seed(seed)

        self.batch_size = batch_size
        self.dataset_path = dataset_path
        self.vocab_size = vocab_size

        self.dataset_train = None
        self.dataset_test = None

        self.trunc_len = trunc_len
        self.workers = workers

        self.model_file = model_file
        self.c_type = c_type

        self.train = []
        self.test = []
        self.seqs = []

    def setup(self, stage=None):
        self.dataset_train = RapppidDataset2(
            self.dataset_path, self.c_type, "train", self.model_file, self.trunc_len
        )
        self.dataset_val = RapppidDataset2(
            self.dataset_path, self.c_type, "val", self.model_file, self.trunc_len
        )
        self.dataset_test = RapppidDataset2(
            self.dataset_path, self.c_type, "test", self.model_file, self.trunc_len
        )

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            num_workers=self.workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            num_workers=self.workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            num_workers=self.workers,
            shuffle=False,
        )
