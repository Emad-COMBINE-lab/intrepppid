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
from random import sample
from pathlib import Path
from functools import cache
from collections import defaultdict


class IntrepppidDataset2(Dataset):
    def __init__(
        self,
        dataset_path,
        c_type,
        split,
        model_file,
        trunc_len=1000,
        sos=False,
        eos=False,
        negative_omid=False,
    ):
        super().__init__()

        self.trunc_len = trunc_len
        self.dataset_path = dataset_path
        self.c_type = c_type
        self.split = split

        if self.split in ["test", "val"]:
            self.sampling = False
        else:
            self.sampling = True

        self.sos = sos
        self.eos = eos

        self.spp = sp.SentencePieceProcessor(model_file=model_file)

        self.negative_omid = negative_omid

        if self.negative_omid:
            with tb.open_file(self.dataset_path) as dataset:
                self.all_omids = [x[0] for x in dataset.root.orthologs.iterrows()]

    @staticmethod
    def static_encode(
        trunc_len: int,
        spp,
        seq: str,
        sp: bool = True,
        pad: bool = True,
        sampling=True,
        sos=False,
        eos=False,
    ):
        seq = seq[:trunc_len]

        if sp:
            toks = spp.encode(seq, enable_sampling=sampling, alpha=0.1, nbest_size=-1)

            if sos:
                toks = [spp.bos_id()] + toks

            if eos:
                toks = toks + [spp.eos_id()]

            toks = np.array(toks)

        else:
            toks = encode_seq(seq)

        if pad:
            pad_len = trunc_len - len(toks)
            toks = np.pad(toks, (0, pad_len), "constant")

        return toks

    def encode(self, seq: str, sp: bool = True, pad: bool = True):
        return self.static_encode(
            self.trunc_len, self.spp, seq, sp, pad, self.sampling, self.sos, self.eos
        )

    @cache
    def get_sequence(self, name: str):
        with tb.open_file(self.dataset_path) as dataset:
            seq = dataset.root.sequences.read_where(f'name=="{name}"')[0][1].decode(
                "utf8"
            )

        return seq

    @cache
    def get_omid_members(self, omid: int):
        with tb.open_file(self.dataset_path) as dataset:
            rows = [
                x[1].decode("utf8")
                for x in dataset.root.orthologs.read_where(f"ortholog_group_id=={omid}")
            ]

        return rows

    def get_omid_member(self, omid: int):
        rows = self.get_omid_members(omid)

        seq = ""
        i = 0
        while len(seq) == 0 or i > 5:
            rand_member = sample(rows, 1)[0]
            seq = self.encode(self.get_sequence(rand_member), sp=True, pad=True)
            i += 1

        return seq

    def __getitem__(self, idx):
        with tb.open_file(self.dataset_path) as dataset:
            p1, p2, omid_pid, omid_id, label = dataset.root["interactions"][
                f"c{self.c_type}"
            ][f"c{self.c_type}_{self.split}"][idx]

        p1 = p1.decode("utf8")
        p2 = p2.decode("utf8")
        omid_pid = omid_pid.decode("utf8")
        omid_id = omid_id

        p1_seq = self.encode(self.get_sequence(p1), sp=True, pad=True)
        p2_seq = self.encode(self.get_sequence(p2), sp=True, pad=True)
        omid1_seq = self.encode(self.get_sequence(omid_pid), sp=True, pad=True)
        omid2_seq = self.get_omid_member(omid_id)

        if self.negative_omid:
            neg_omid_id = sample(self.all_omids, 1)[0]
            omid_neg_seq = self.get_omid_member(neg_omid_id)
            omid_neg_seq = torch.tensor(omid_neg_seq).long()

        p1_seq = torch.tensor(p1_seq).long()
        p2_seq = torch.tensor(p2_seq).long()
        omid1_seq = torch.tensor(omid1_seq).long()
        omid2_seq = torch.tensor(omid2_seq).long()
        label = torch.tensor(label).long()

        if self.negative_omid:
            return p1_seq, p2_seq, omid1_seq, omid2_seq, omid_neg_seq, label
        else:
            return p1_seq, p2_seq, omid1_seq, omid2_seq, label

    def __len__(self):
        with tb.open_file(self.dataset_path) as dataset:
            l = len(
                dataset.root["interactions"][f"c{self.c_type}"][
                    f"c{self.c_type}_{self.split}"
                ]
            )
        return l


class IntrepppidDataModule2(pl.LightningDataModule):
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
        sos: bool,
        eos: bool,
        negative_omid: bool = False,
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

        self.sos = sos
        self.eos = eos

        self.negative_omid = negative_omid

    def setup(self, stage=None):
        self.dataset_train = IntrepppidDataset2(
            self.dataset_path,
            self.c_type,
            "train",
            self.model_file,
            self.trunc_len,
            self.sos,
            self.eos,
            self.negative_omid,
        )
        self.dataset_val = IntrepppidDataset2(
            self.dataset_path,
            self.c_type,
            "val",
            self.model_file,
            self.trunc_len,
            self.sos,
            self.eos,
            self.negative_omid,
        )
        self.dataset_test = IntrepppidDataset2(
            self.dataset_path,
            self.c_type,
            "test",
            self.model_file,
            self.trunc_len,
            self.sos,
            self.eos,
            self.negative_omid,
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


class IntrepppidDataset(Dataset):
    def __init__(
        self,
        dataset_path: Path,
        c_type: int,
        split: str,
        model_file: Path,
        trunc_len: int = 1000,
        sos: bool = False,
        eos: bool = False,
        negative_omid: bool = False,
    ):
        """
        Builds a PyTorch dataset from an HDF5 dataset in the INTREPPPID format.

        :param dataset_path: The path to the HDF5 in the INTREPPPID format.
        :param c_type: The C-type pairs to use in the dataset.
        :param split: Which split the data should come from (`i.e.`: ``train``, ``test``, ``val``)
        :param model_file: The path to the SentencePiece model.
        :param trunc_len: The length at which to truncate the amino acid sequence.
        :param sos: Boolean indicating whether a start-of-sequence token should be used.
        :param eos: Boolean indicating whether an end-of-sequence token should be used.
        :param negative_omid: Boolean indicating whether to return a negative example for the orthologous locality task.
        """
        super().__init__()

        self.trunc_len = trunc_len
        self.dataset_path = dataset_path
        self.c_type = c_type
        self.split = split

        if self.split in ["test", "val"]:
            self.sampling = False
        else:
            self.sampling = True

        self.sos = sos
        self.eos = eos

        self.spp = sp.SentencePieceProcessor(model_file=model_file)

        self.negative_omid = negative_omid

        self.interactions = []
        self.sequences = {}
        self.omid_members = defaultdict(lambda: [])

        with tb.open_file(self.dataset_path) as dataset:
            print("loading interactions...")
            for row in dataset.root["interactions"][f"c{self.c_type}"][
                f"c{self.c_type}_{self.split}"
            ]:
                p1, p2, omid_pid, omid_id, label = (
                    row["protein_id1"].decode("utf8"),
                    row["protein_id2"].decode("utf8"),
                    row["omid_protein_id"].decode("utf8"),
                    row["omid_id"],
                    row["label"],
                )
                self.interactions.append((p1, p2, omid_pid, omid_id, label))

            print("loading sequences...")
            for row in dataset.root.sequences.iterrows():
                name = row["name"].decode("utf8")
                sequence = row["sequence"].decode("utf8")
                self.sequences[name] = sequence

            print("loading orthogroups...")
            for row in dataset.root.orthologs.iterrows():
                ortholog_group_id = row["ortholog_group_id"]
                protein_id = row["protein_id"].decode("utf8")
                self.omid_members[ortholog_group_id].append(protein_id)

    @staticmethod
    def static_encode(
        trunc_len: int,
        spp,
        seq: str,
        sp: bool = True,
        pad: bool = True,
        sampling: bool = True,
        sos: bool = False,
        eos: bool = False,
    ):
        """
        Encodes an amino-acid sequence using the SentencePiece model, pads, and adds start/end-of-sequence tokens.

        This is the static method, which doesn't require this class to be instantiated.

        :param trunc_len: The length at which to truncate the amino acid sequence.
        :param spp: The SentencePiece model to use to encode the sample.
        :param seq: The amino acid sequence to be encoded.
        :param sp: A boolean which indicates whether to encode the sequence with SentencePiece
        :param pad: Whether to pad up to the trunc len (``True``) or not (``False``). Pads on the right with zeroes.
        :param sampling: Whether to randomly sample tokens (``True``) or not (``False``). See the `SentencePiece library <https://github.com/google/sentencepiece>`_/`manuscript <https://arxiv.org/abs/1808.06226>`_ for details.
        :param sos: Boolean indicating whether to add a start-of-sequence token (``True``) or not (``False``)
        :param eos: Boolean indicating whether to add an end-of-sequence token (``True``) or not (``False``)
        """
        seq = seq[:trunc_len]

        if sp:
            toks = spp.encode(seq, enable_sampling=sampling, alpha=0.1, nbest_size=-1)

            if sos:
                toks = [spp.bos_id()] + toks

            if eos:
                toks = toks + [spp.eos_id()]

            toks = np.array(toks)

        else:
            toks = encode_seq(seq)

        if pad:
            pad_len = trunc_len - len(toks)
            toks = np.pad(toks, (0, pad_len), "constant")

        return toks

    def encode(self, seq: str, sp: bool = True, pad: bool = True):
        """
        Encodes an amino-acid sequence using the SentencePiece model, pads, and adds start/end-of-sequence tokens.

        :param seq: The amino acid sequence to be encoded.
        :param sp: A boolean which indicates whether to encode the sequence with SentencePiece.
        :param pad: Whether to pad up to the trunc len (``True``) or not (``False``). Pads on the right with zeroes.
        """
        return self.static_encode(
            self.trunc_len, self.spp, seq, sp, pad, self.sampling, self.sos, self.eos
        )

    @cache
    def get_sequence(self, name: str):
        """
        Get the amino-acid sequence of a protein in the dataset given its Uniprot Accession.

        :param name: The Uniprot Accession
        """
        with tb.open_file(self.dataset_path) as dataset:
            seq = dataset.root.sequences.read_where(f'name=="{name}"')[0][1].decode(
                "utf8"
            )

        return seq

    @cache
    def get_omid_members(self, omid: int):
        """
        Get all the Uniprot Accessions of the proteins sharing in the same specified OMA group in this dataset.

        :param omid: The OMA ID.
        """
        rows = self.omid_members[omid]

        return rows

    def get_omid_member(self, omid: int):
        """
        Get a random Uniprot Accessions of a protein in the specified OMA group in this dataset.

        :param omid: The OMA ID.
        """
        rows = self.get_omid_members(omid)

        rand_member = ""
        seq = None
        i = 0
        while seq is None and i < 5:
            rand_member = sample(rows, 1)[0]
            try:
                seq = self.sequences[rand_member]
            except KeyError:
                print(rand_member)
            i += 1

        if seq is None:
            seq = "M"

        seq = self.encode(seq, sp=True, pad=True)

        return seq

    def __getitem__(self, idx: int):
        """
        Get a sample from the dataset by its index.

        This returns:

        ``p1_seq, p2_seq, omid1_seq, omid2_seq, omid_neg_seq, label``

        if ``negative_omid`` is ``True``. Otherwise:

        ``p1_seq, p2_seq, omid1_seq, omid2_seq, label``

        Where ``p1_seq`` and ``p2_seq`` are two amino acid sequences whose interaction status is indicated by ``label``,
        ``omid1_seq`` is the anchor protein sequence for the orthologous locality task, ``omid2_seq`` is the positive
        protein sequence, and ``omid_neg_seq`` is the negative protein sequence.

        Negative proteins are randomly sampled.

        :param idx: The index of the sample to return.
        """
        p1, p2, omid_pid, omid_id, label = self.interactions[idx]

        p1_seq = self.encode(self.sequences[p1], sp=True, pad=True)
        p2_seq = self.encode(self.sequences[p2], sp=True, pad=True)
        try:
            omid1_seq = self.encode(self.sequences[omid_pid], sp=True, pad=True)
            omid2_seq = self.get_omid_member(omid_id)
        except KeyError:
            print("Orthologue not found.", omid_pid)
            omid1_seq = p1_seq
            omid2_seq = p1_seq

        if self.negative_omid:
            neg_omid_id = sample(self.omid_members.keys(), 1)[0]
            omid_neg_seq = self.get_omid_member(neg_omid_id)
            omid_neg_seq = torch.tensor(omid_neg_seq).long()

        p1_seq = torch.tensor(p1_seq).long()
        p2_seq = torch.tensor(p2_seq).long()
        omid1_seq = torch.tensor(omid1_seq).long()
        omid2_seq = torch.tensor(omid2_seq).long()
        label = torch.tensor(label).long()

        if self.negative_omid:
            return p1_seq, p2_seq, omid1_seq, omid2_seq, omid_neg_seq, label
        else:
            return p1_seq, p2_seq, omid1_seq, omid2_seq, label

    def __len__(self):
        """
        Get the number of samples in the dataset.
        """
        with tb.open_file(self.dataset_path) as dataset:
            l = len(
                dataset.root["interactions"][f"c{self.c_type}"][
                    f"c{self.c_type}_{self.split}"
                ]
            )
        return l


class IntrepppidDataModule(pl.LightningDataModule):
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
        sos: bool,
        eos: bool,
        negative_omid: bool = False,
    ):
        """
        A `PyTorch Lightning <https://lightning.ai/docs/pytorch/stable/>`_ `Data Module <https://lightning.ai/docs/pytorch/1.9.3/api/pytorch_lightning.core.LightningDataModule.html>`_ for INTREPPPID datasets.

        :param batch_size: The size of the batches for the Data Module to generate
        :param dataset_path: The path to the HDF5 in the INTREPPPID format.
        :param c_type: The C-type pairs to use in the dataset.
        :param trunc_len: The length at which to truncate the amino acid sequence.
        :param workers: The number of CPU processes used to load data.
        :param vocab_size: The number of tokens in the SentencePiece vocabular.
        :param model_file: The path to the SentencePiece model.
        :param seed: The random seed to use for sampling SentencePiece tokens.
        :param sos: Boolean indicating whether to add a start-of-sequence token (``True``) or not (``False``)
        :param eos: Boolean indicating whether to add an end-of-sequence token (``True``) or not (``False``)
        :param negative_omid: Boolean indicating whether to return a negative example for the orthologous locality task.
        """
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

        self.sos = sos
        self.eos = eos

        self.negative_omid = negative_omid

    def setup(self, stage=None):
        """
        Instantiate the internal datasets.

        :param stage: Specifies the stage of training the model (must be one of ``train``, ``val``, or ``test``).
        """
        self.dataset_train = IntrepppidDataset(
            self.dataset_path,
            self.c_type,
            "train",
            self.model_file,
            self.trunc_len,
            self.sos,
            self.eos,
            self.negative_omid,
        )
        self.dataset_val = IntrepppidDataset(
            self.dataset_path,
            self.c_type,
            "val",
            self.model_file,
            self.trunc_len,
            self.sos,
            self.eos,
            self.negative_omid,
        )
        self.dataset_test = IntrepppidDataset(
            self.dataset_path,
            self.c_type,
            "test",
            self.model_file,
            self.trunc_len,
            self.sos,
            self.eos,
            self.negative_omid,
        )

    def train_dataloader(self):
        """
        Returns the dataloader for the training set.
        """
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            num_workers=self.workers,
            shuffle=True,
        )

    def val_dataloader(self):
        """
        Returns the dataloader for the validation set.
        """
        return DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            num_workers=self.workers,
            shuffle=False,
        )

    def test_dataloader(self):
        """
        Returns the dataloader for the test set.
        """
        return DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            num_workers=self.workers,
            shuffle=False,
        )
