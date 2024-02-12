# INTREPPPID

[![DOI](https://zenodo.org/badge/748346659.svg)](https://zenodo.org/doi/10.5281/zenodo.10652231)

***IN**corporating **TR**iplet **E**rror for **P**redicting **P**rotein-**P**rotein **I**nteractions using **D**eep Learning*

---


INTREPPPID is a deep learning model for predicting protein interactions. 
It's especially good at making prediction on species other than those it was trained on (cross-species prediction).

## How to Use INTREPPPID

Here are some quick highlights, but be sure to [read the documentation](https://emad-combine-lab.github.io/intrepppid/) for more details!

### Installing

The easiest way to install PPI Origami is to use [pip](https://pip.pypa.io/en/stable/>) to retrieve the PPI Origami
release from [PyPI](https://pypi.org/project/ppi-origami>).

```bash
pip install intrepppid
```

Alternatively, clone the repository and use [poetry](https://python-poetry.org/) to install the dependencies

```bash
git clone https://github.com/jszym/intrepppid
cd intreppid
poetry install
```

### Training Models

To train INTREPPPID, simply use the `train e2e_rnn_triplet` command like so:

```bash
intrepppid train e2e_rnn_triplet DATASET.h5 spm.model 3 100 80 --seed 3927704 --vocab_size 250 --trunc_len 1500 --embedding_size 64 --rnn_num_layers 2 --rnn_dropout_rate 0.3 --variational_dropout false --bi_reduce last --workers 4 --embedding_droprate 0.3 --do_rate 0.3 --log_path logs/e2e_rnn_triplet --beta_classifier 2 --use_projection false --optimizer_type ranger21_xx --lr 1e-2
```

### Documentation

Be sure to [read the documentation]((https://emad-combine-lab.github.io/intrepppid/)) for more details.

## License

INTREPPPID

***IN**corporating **TR**iplet **E**rror for **P**redicting **P**rotein-**P**rotein **I**nteractions using **D**eep Learning*

Copyright (C) 2023  Joseph Szymborski

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
