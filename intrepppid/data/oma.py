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
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pandas as pd
import sentencepiece as sp
import numpy as np
from intrepppid.data.utils import encode_seq


class OmaTripletDataset(Dataset):
    def __init__(
        self, triplets_path, seqs_path, model_file, sampling, split, trunc_len=1500
    ):
        super().__init__()

        self.trunc_len = trunc_len
        self.seqs = (
            pd.read_csv(seqs_path).drop_duplicates(keep="first").set_index("upkb_ac")
        )
        self.triplets = pd.read_csv(triplets_path)
        one_tenth = len(self.triplets) // 10

        if split == "train":
            self.triplets = self.triplets[: one_tenth * 8].sample(frac=1.0)
        elif split == "val":
            self.triplets = self.triplets[one_tenth * 8 : one_tenth * 9]
        elif split == "test":
            self.triplets = self.triplets[one_tenth * 9 :]
        else:
            raise ValueError(f'split must be train, val, or test. got "{split}"')

        self.sampling = sampling

        self.spp = sp.SentencePieceProcessor(model_file=str(model_file))

    @staticmethod
    def static_encode(
        trunc_len: int, spp, seq: str, sp: bool = True, pad: bool = True, sampling=True
    ):
        try:
            seq = seq[:trunc_len]
        except TypeError:
            print(seq)

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

    def __getitem__(self, idx):
        anchor = self.triplets.iloc[idx].anchor
        positive = self.triplets.iloc[idx].positive
        negative = self.triplets.iloc[idx].negative

        anchor_seq = self.encode(self.seqs.loc[anchor].seq)
        positive_seq = self.encode(self.seqs.loc[positive].seq)
        negative_seq = self.encode(self.seqs.loc[negative].seq)

        return anchor_seq, positive_seq, negative_seq

    def __len__(self):
        return len(self.triplets)


class OmaTripletDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        dataset_folder: Path,
        seqs_path: Path,
        model_path: Path,
        num_workers: int,
        trunc_len: int,
    ):
        super().__init__()
        self.df_val = None
        self.df_test = None
        self.df_train = None
        self.batch_size = batch_size
        self.dataset_folder = dataset_folder
        self.seqs_path = seqs_path
        self.model_path = model_path
        self.num_workers = num_workers
        self.trunc_len = trunc_len

    def setup(self, stage: str):
        self.df_train = OmaTripletDataset(
            self.dataset_folder,
            self.seqs_path,
            self.model_path,
            True,
            "train",
            self.trunc_len,
        )
        self.df_test = OmaTripletDataset(
            self.dataset_folder,
            self.seqs_path,
            self.model_path,
            False,
            "test",
            self.trunc_len,
        )
        self.df_val = OmaTripletDataset(
            self.dataset_folder,
            self.seqs_path,
            self.model_path,
            False,
            "val",
            self.trunc_len,
        )

    def train_dataloader(self):
        return DataLoader(
            self.df_train, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.df_val, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.df_test, batch_size=self.batch_size, num_workers=self.num_workers
        )
