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

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Type

from mava.callbacks import Callback
from mava.components import Component
from mava.core_jax import SystemBuilder
from mava.utils.loggers import MavaLogger
from mava.utils.loggers.eval_json_logger import JSONLogger


@dataclass
class LoggerConfig:
    logger_factory: Optional[Callable[[str], MavaLogger]] = None
    logger_config: Optional[Any] = None
    system_name: str = "system"


class Logger(Component):
    def __init__(
        self,
        config: LoggerConfig = LoggerConfig(),
    ):
        """Component creates executor, trainer, and evaluator loggers.

        Args:
            config: LoggerConfig.
        """
        self.config = config

    def on_building_executor_logger(self, builder: SystemBuilder) -> None:
        """Create and store either the executor or evaluator logger.

        Args:
            builder: SystemBuilder.

        Returns:
            None.
        """
        logger_config = self.config.logger_config if self.config.logger_config else {}
        # is_evaluator set by builder
        name = "executor" if not builder.store.is_evaluator else "evaluator"

        if self.config.logger_config and name in self.config.logger_config:
            logger_config = self.config.logger_config[name]

        # executor_id set by builder
        builder.store.executor_logger = self.config.logger_factory(  # type: ignore
            builder.store.executor_id, **logger_config
        )

        # Add json logger to evaluator process only when performing evaluation
        # at fixed intervals and for fixed durations.
        if (
            name == "evaluator"
            and builder.store.global_config.evaluation_interval is not None
        ):
            # Instantiate an environment from a factory to get access to the
            # environment name and task name
            (
                temp_environment,
                environment_name_dict,
            ) = builder.store.global_config.environment_factory(evaluation=False)
            del temp_environment

            builder.store.eval_json_logger = JSONLogger(
                experiment_path=builder.store.global_config.experiment_path,
                random_seed=builder.store.global_config.seed,
                env_name=environment_name_dict["environment_name"],
                task_name=environment_name_dict["task_name"],
                system_name=self.config.system_name,
            )

    def on_building_trainer_logger(self, builder: SystemBuilder) -> None:
        """Create and store the trainer logger.

        Args:
            builder: SystemBuilder.

        Returns:
            None.
        """
        logger_config = self.config.logger_config if self.config.logger_config else {}
        name = "trainer"
        if self.config.logger_config and name in self.config.logger_config:
            logger_config = self.config.logger_config[name]

        # trainer_id set by builder
        builder.store.trainer_logger = self.config.logger_factory(  # type: ignore
            builder.store.trainer_id, **logger_config
        )

    @staticmethod
    def name() -> str:
        """Static method that returns component name."""
        return "logger"

    @staticmethod
    def required_components() -> List[Type[Callback]]:
        """List of other Components required in the system for this Component to function.

        None required.

        Returns:
            List of required component classes.
        """
        return []
