# python3
# Copyright 2021 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Execution components for system builders"""

from mava.components.jax.building.base import BaseTrainerProcess
from mava.core_jax import SystemBuilder


class TrainerProcess(BaseTrainerProcess):
    def on_building_trainer(self, builder: SystemBuilder) -> None:
        """_summary_

        Args:
            builder : _description_
        """
        # builder.trainer_fn = Trainer(builder.components)