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

from intrepppid.classifier.barlow import make_classifier_barlow


def get_classifier(model_name: str, **kwargs):
    if model_name == "barlow":
        required_args = {
            "barlow_hyperparams_path",
            "barlow_checkpoint_path",
            "embedding_droprate",
            "do_rate",
            "steps_per_epoch",
            "num_epochs",
            "fine_tune_epochs",
        }

        arg_diff = set(kwargs) - required_args

        if len(arg_diff) != 0:
            raise ValueError(f"{model_name} is missing arguments: {arg_diff}")

        net = make_classifier_barlow(
            barlow_hyperparams_path=kwargs["barlow_hyperparams_path"],
            barlow_checkpoint_path=kwargs["barlow_checkpoint_path"],
            embedding_droprate=kwargs["embedding_droprate"],
            do_rate=kwargs["do_rate"],
            steps_per_epoch=kwargs["steps_per_epoch"],
            num_epochs=kwargs["num_epochs"],
            fine_tune_epochs=kwargs["fine_tune_epochs"],
        )

        return net
