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

from intrepppid.classifier.head.mlp import MLPHead
from typing import Dict, Any
from torch import nn


def get_head(hyperparams: Dict[str, Any]) -> nn.Module:
    architecture = hyperparams['architecture']
    del hyperparams[architecture]

    if architecture == "MLPHead":
        return MLPHead(**architecture)
    else:
        raise ValueError("Unexpected architecture. Must be one of 'MLPHead'.")
