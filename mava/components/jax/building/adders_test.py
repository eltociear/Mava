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

"""Tests for config class for Jax-based Mava systems"""

from types import SimpleNamespace

import pytest

from mava.components.jax.building import adders
from mava.systems.jax.system import System
from mava.testing.building import mocks


class TestSystem(System):
    def design(self) -> SimpleNamespace:
        """Mock system design with zero components.

        Returns:
            system callback components
        """
        executor_adder = adders.ParallelSequenceAdder
        data_server_adder = adders.ParallelSequenceAdderSignature
        components = SimpleNamespace(
            data_server=mocks.MockDataServer,
            data_server_adder=data_server_adder,
            parameter_server=mocks.MockParameterServer,
            parameter_client=mocks.MockParameterClient,
            logger=mocks.MockLogger,
            executor=mocks.MockExecutor,
            executor_adder=executor_adder,
            executor_environment_loop=mocks.MockExecutorEnvironmentLoop,
            trainer=mocks.MockTrainer,
            trainer_dataset=mocks.MockTrainerDataset,
            distributor=mocks.MockDistributor,
        )
        return components


@pytest.fixture
def test_system() -> System:
    """Dummy system with zero components."""
    return TestSystem()


def test_adders(
    test_system: System,
) -> None:
    """Test if system builder instantiates processes as expected."""
    test_system.launch(num_executors=1, nodes_on_gpu=["process"])