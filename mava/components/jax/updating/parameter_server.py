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

"""Parameter server component for Mava systems."""
from dataclasses import dataclass

from mava.callbacks import Callback
from mava.core_jax import SystemBuilder


@dataclass
class ParameterServerConfig:
    parameter_server_param: str = "Testing"
    Second_var: str = "Testing2"


class DefaultParameterServer(Callback):
    def __init__(
        self,
        config: ParameterServerConfig = ParameterServerConfig(),
    ) -> None:
        """Mock system component."""
        self.config = config

    def on_building_parameter_server(self, builder: SystemBuilder) -> None:
        """_summary_"""
        builder.attr.system_parameter_server = self.config.parameter_server_param

    @property
    def name(self) -> str:
        """Component type name, e.g. 'dataset' or 'executor'."""
        return "parameter_server"