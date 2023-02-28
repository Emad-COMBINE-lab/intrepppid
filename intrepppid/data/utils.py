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

from random import randint
from typing import List


def get_aa_code(aa: str) -> int:
    """
    Translate an amino acid into an integer. Wobble pairs are randomly assigned an amino acid from
    the possible amino acids it represents.

    :param aa: A character representing an amino acid
    :return: An integer representing the same amino acid as the one inputted.
    """
    # Codes based on IUPAC-IUB
    # https://web.expasy.org/docs/userman.html#AA_codes

    aas = [
        "PAD",
        "A",
        "R",
        "N",
        "D",
        "C",
        "Q",
        "E",
        "G",
        "H",
        "I",
        "L",
        "K",
        "M",
        "F",
        "P",
        "S",
        "T",
        "W",
        "Y",
        "V",
        "O",
        "U",
    ]
    wobble_aas = {
        "B": ["D", "N"],
        "Z": ["Q", "E"],
        "X": [
            "A",
            "R",
            "N",
            "D",
            "C",
            "Q",
            "E",
            "G",
            "H",
            "I",
            "L",
            "K",
            "M",
            "F",
            "P",
            "S",
            "T",
            "W",
            "Y",
            "V",
        ],
    }

    if aa in aas:
        return aas.index(aa)

    elif aa in ["B", "Z", "X"]:
        # Wobble
        idx = randint(0, len(wobble_aas[aa]) - 1)
        return aas.index(wobble_aas[aa][idx])


def encode_seq(seq) -> List[int]:
    """
    Converts a string of characters representing amino acids into a list of integers.

    :param seq: A string of characters representing amino acids.
    :return: A list of integers representing these integers.
    """
    return [get_aa_code(aa) for aa in seq]
