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
import abc
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import dm_env

from mava import specs
from mava.callbacks import Callback
from mava.core_jax import SystemBuilder
from mava.environment_loop import ParallelEnvironmentLoop


@dataclass
class ExecutorEnvironmentLoopConfig:
    environment_factory: Optional[Callable[[bool], dm_env.Environment]] = None
    environment_kwargs: Dict = {}


class ExecutorEnvironmentLoop(Callback):
    def __init__(
        self, config: ExecutorEnvironmentLoopConfig = ExecutorEnvironmentLoopConfig()
    ):
        """[summary]"""
        self.config = config

    def on_building_init(self, builder: SystemBuilder) -> None:
        """[summary]"""
        builder.attr.environment_spec = specs.MAEnvironmentSpec(
            self.config.environment_factory(evaluation=False)  # type: ignore
        )

    def on_building_executor_environment(self, builder: SystemBuilder) -> None:
        """_summary_

        Args:
            builder : _description_
        """
        builder.attr.executor_environment = self.config.environment_factory(
            evaluation=False  # type: ignore
        )

    @abc.abstractmethod
    def on_building_executor_environment_loop(self, builder: SystemBuilder) -> None:
        """[summary]"""


class ParallelExecutorEnvironmentLoop(ExecutorEnvironmentLoop):
    def on_building_executor_environment_loop(self, builder: SystemBuilder) -> None:
        """_summary_

        Args:
            builder : _description_
        """
        executor_environment_loop = ParallelEnvironmentLoop(
            builder.attr.executor_environment,
            builder.attr.executor_fn,
            logger=builder.attr.executor_logger,
            **self.config.environment_kwargs,
        )
        if builder._executor_id == "evaluator":
            builder.attr.system_evaluator = executor_environment_loop
        else:
            builder.attr.system_executor = executor_environment_loop