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

"""Integration test of the Trainer for Jax-based Mava"""

import os
import signal
import time

import jax.numpy as jnp
import pytest

from mava.systems.jax import System
from tests.jax.integration.mock_systems import (
    mock_system_multi_process,
    mock_system_multi_thread,
    mock_system_single_process,
)


@pytest.fixture
def test_system_sp() -> System:
    """A single process built system"""
    return mock_system_single_process()


@pytest.fixture
def test_system_mp() -> System:
    """A multi process built system"""
    return mock_system_multi_process()


@pytest.fixture
def test_system_mt() -> System:
    """A multi thread built system"""
    return mock_system_multi_thread()


def test_trainer_single_process(test_system_sp: System) -> None:
    """Test if the trainer instantiates processes as expected."""
    (
        data_server,
        parameter_server,
        executor,
        evaluator,
        trainer,
    ) = test_system_sp._builder.store.system_build

    for _ in range(5):
        executor.run_episode()

    # Before run step method
    for net_key in trainer.store.networks["networks"].keys():
        mu = trainer.store.opt_states[net_key][1][0][-1]  # network
        for categorical_value_head in mu.values():
            assert jnp.all(categorical_value_head["b"] == 0)
            assert jnp.all(categorical_value_head["w"] == 0)

    trainer.step()

    # After run step method
    for net_key in trainer.store.networks["networks"].keys():
        mu = trainer.store.opt_states[net_key][1][0][-1]
        for categorical_value_head in mu.values():
            assert not jnp.all(categorical_value_head["b"] == 0)
            assert not jnp.all(categorical_value_head["w"] == 0)


def test_trainer_multi_thread(test_system_mt: System) -> None:
    """Test if the trainer instantiates processes as expected."""
    # Disable the run of the trainer node
    (trainer_node,) = test_system_mt._builder.store.program._program._groups["trainer"]
    trainer_node.disable_run()

    # launch the system
    test_system_mt.launch()
    time.sleep(10)  # wait till the executor has run

    trainer = trainer_node._construct_instance()

    # Before run step function
    for net_key in trainer.store.networks["networks"].keys():
        mu = trainer.store.opt_states[net_key][1][0][-1]  # network
        for categorical_value_head in mu.values():
            assert jnp.all(categorical_value_head["b"] == 0)
            assert jnp.all(categorical_value_head["w"] == 0)

    # Step function
    trainer.step()

    # Check that the trainer update the network
    for net_key in trainer.store.networks["networks"].keys():
        mu = trainer.store.opt_states[net_key][1][0][-1]
        for categorical_value_head in mu.values():
            assert not jnp.all(categorical_value_head["b"] == 0)
            assert not jnp.all(categorical_value_head["w"] == 0)


def test_trainer_multi_process(test_system_mp: System) -> None:
    """Test if the trainer instantiates processes as expected."""
    # Disable the run of the trainer node
    (trainer_node,) = test_system_mp._builder.store.program._program._groups["trainer"]
    trainer_node.disable_run()

    # pid to help stop the launcher once the test ends
    pid = os.getpid()
    # launch the system
    test_system_mp.launch()

    time.sleep(10)  # wait till the executor has run

    trainer = trainer_node._construct_instance()

    # Before run step function
    for net_key in trainer.store.networks["networks"].keys():
        mu = trainer.store.opt_states[net_key][1][1]  # network
        for categorical_value_head in mu.values():
            assert jnp.all(categorical_value_head["b"] == 0)
            assert jnp.all(categorical_value_head["w"] == 0)

    # Step function
    trainer.step()

    # Check that the trainer update the network
    for net_key in trainer.store.networks["networks"].keys():
        mu = trainer.store.opt_states[net_key][1][1]
        for categorical_value_head in mu.values():
            assert not jnp.all(categorical_value_head["b"] == 0)
            assert not jnp.all(categorical_value_head["w"] == 0)

    # stop the launcher
    os.kill(pid, signal.SIGTERM)