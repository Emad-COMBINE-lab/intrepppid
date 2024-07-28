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

from pathlib import Path
from typing import Optional

import lmdb
from torch import Tensor

from intrepppid.data.ppi_oma import IntrepppidDataset
from intrepppid import intrepppid_network
import torch
import sentencepiece as sp
import requests
import tempfile
import gzip
import shutil
import json
import time
import csv


deleted_uniprot_acs = set()


def stream_fasta(fasta_path: Path):
    if str(fasta_path).endswith(".gz"):
        f = gzip.open(str(fasta_path), "rt")
    else:
        f = open(str(fasta_path), "rt")

    sequence = None

    for line in f:
        line = line.strip()
        if line.startswith(">"):
            if sequence is not None and sequence != "":
                yield name, sequence
            name = line[1:]
            sequence = ""
        else:
            sequence += line


def get_uniprot_seq(uniprot_ac: str):

    if uniprot_ac in deleted_uniprot_acs:
        print(f"Failed to get sequence for \"{uniprot_ac}\" from UniProt (it was likely deleted)")
        return None

    time.sleep(1)
    r = requests.get(f'https://rest.uniprot.org/uniprotkb/{uniprot_ac}.fasta')

    if r.status_code == requests.codes.ok:
        seq = ""
        for idx, line in enumerate(r.text.split('\n')):

            # skip header
            if idx == 0:
                continue

            seq += line.strip()

        if seq == "":
            print(f"Failed to get sequence for \"{uniprot_ac}\" from UniProt (it was likely deleted)")
            deleted_uniprot_acs.add(uniprot_ac)
            return None
        else:
            print(f"🆗 Found sequence for \"{uniprot_ac}\" via UniProt")
            return seq
    else:
        print(f"Failed to get sequence for \"{uniprot_ac}\" from UniProt")
        return None


class Infer(object):

    @staticmethod
    def from_csv(interactions_path: Path, sequences_path: Path, weights_path: Path, spm_path: Path, out_path: Path,
                 trunc_len: int = 1500, low_memory: bool = False, db_path: Path = None, dont_populate_db: bool = False,
                 device: str = 'cpu', get_from_uniprot: bool = False):

        spp = sp.SentencePieceProcessor(model_file=spm_path)

        try:
            # Build sequence library
            if low_memory:
                if db_path is None:
                    db_path = tempfile.mkdtemp(prefix="intrepppid_")

                seq_db = lmdb.open(db_path)
                seq_db.set_mapsize(1024 * 1024 * 1024 * 1024)

                if not dont_populate_db:
                    print("Building sequence db...")
                    with seq_db.begin(write=True) as txn:
                        for name, sequence in stream_fasta(sequences_path):

                            embed = IntrepppidDataset.static_encode(trunc_len, spp, sequence).tolist()
                            embed = json.dumps(embed)
                            txn.put(name.encode('utf8'), embed.encode('utf8'))

                def get_embed(name: str) -> Optional[Tensor]:
                    with seq_db.begin() as txn:
                        vec_str = txn.get(name.encode('utf8'))

                    if vec_str is None:

                        if get_from_uniprot:
                            print(f"Sequence for \"{name}\" not found in file, searching UniProt...")
                            sequence = get_uniprot_seq(name)

                            if sequence is None:
                                return None

                            embed = IntrepppidDataset.static_encode(trunc_len, spp, sequence).tolist()
                            embed_str = json.dumps(embed)

                            with seq_db.begin(write=True) as txn:
                                txn.put(name.encode('utf8'), embed_str.encode('utf8'))

                            return torch.tensor(embed)

                        else:
                            print(f"Failed to get embedding from \"{name}\".")
                            return None
                    else:
                        return torch.tensor(json.loads(vec_str.decode('utf8')))
            else:
                embeddings = {}

                for name, sequence in stream_fasta(sequences_path):
                    embed = IntrepppidDataset.static_encode(trunc_len, spp, sequence)
                    embeddings[name] = embed

                def get_embed(name: str) -> Optional[Tensor]:
                    if get_from_uniprot and name not in embeddings:
                        print(f"Sequence for \"{name}\" not found in file, searching UniProt...")
                        sequence = get_uniprot_seq(name)

                        if sequence is None:
                            return None

                        embed = IntrepppidDataset.static_encode(trunc_len, spp, sequence).tolist()

                        embeddings[name] = embed

                        return torch.tensor(embed)

                    if name not in embeddings:
                        return None

                    return torch.tensor(embeddings[name])

            # Load INTREPPPID model

            net = intrepppid_network(0, use_projection=True)
            net.eval()

            chkpt = torch.load(weights_path)

            net.load_state_dict(chkpt['state_dict'])

            net.to(device)

            # Infer pairs

            with open(out_path, 'w') as f_out:
                csv_writer = csv.DictWriter(f_out, fieldnames=['itx_id', 'probability'])

                if interactions_path.endswith('.gz'):
                    opener = gzip.open
                    mode = "rt"
                else:
                    opener = open
                    mode = "r"

                with opener(interactions_path, mode) as f_in:
                    fieldnames = ['itx_id', 'id_a', 'id_b']
                    csv_reader = csv.DictReader(f_in, fieldnames=fieldnames)

                    intxn = 0

                    for row in csv_reader:
                        itx_id = row['itx_id']

                        embed_a = get_embed(row['id_a'])
                        embed_b = get_embed(row['id_b'])

                        if embed_a is None or embed_b is None:
                            missing_ids = ""
                            if embed_a is None:
                                missing_ids += row['id_a'] + " "
                            if embed_b is None:
                                missing_ids += row['id_b']

                            print(f"💣 Can't compute pair id: {itx_id} (\"{row['id_a']}\", \"{row['id_b']}\")")
                            print(f"\tMissing sequence in database for IDs: {missing_ids}")
                            continue
                        else:
                            intxn += 1

                        # TODO: Batch inference.
                        # For now, use unsqueeze to make batch of 1

                        embed_a = embed_a.unsqueeze(0).to(device)
                        embed_b = embed_b.unsqueeze(0).to(device)

                        logits = net(embed_a, embed_b)

                        probability = torch.sigmoid(logits)
                        probability = probability.detach().cpu().numpy().tolist()[0][0]

                        csv_writer.writerow({'itx_id': itx_id, 'probability': probability})

        finally:
            if low_memory and db_path is None:
                shutil.rmtree(db_path)

