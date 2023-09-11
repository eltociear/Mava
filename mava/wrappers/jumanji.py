# Copyright 2022 InstaDeep Ltd. All rights reserved.
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

from typing import Tuple

import chex
import jax.numpy as jnp
from jumanji import specs
from jumanji.environments.routing.robot_warehouse import (
    Observation,
    RobotWarehouse,
    State,
)
from jumanji.types import TimeStep

from mava.wrappers.base import MavaWrapper


class RwareMultiAgentWrapper(MavaWrapper):
    """Multi-agent wrapper for the Robotic Warehouse environment."""

    def __init__(self, env: RobotWarehouse):
        super().__init__(env)
        # not sure if there's a better way to do this typing?
        self._env: RobotWarehouse

    @property
    def num_agents(self) -> int:
        return self._env.num_agents

    @property
    def time_limit(self) -> int:
        return self._env.time_limit

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep]:
        """Reset the environment. Updates the step count."""
        state, timestep = self._env.reset(key)
        timestep.observation = Observation(
            agents_view=timestep.observation.agents_view,
            action_mask=timestep.observation.action_mask,
            step_count=jnp.repeat(timestep.observation.step_count, self._env.num_agents),
        )
        return state, timestep

    def step(self, state: State, action: chex.Array) -> Tuple[State, TimeStep]:
        """Step the environment. Updates the step count."""
        state, timestep = self._env.step(state, action)
        timestep.observation = Observation(
            agents_view=timestep.observation.agents_view,
            action_mask=timestep.observation.action_mask,
            step_count=jnp.repeat(timestep.observation.step_count, self._env.num_agents),
        )
        return state, timestep

    def observation_spec(self) -> specs.Spec[Observation]:
        """Specification of the observation of the `RobotWarehouse` environment."""
        step_count = specs.BoundedArray(
            (self._env.num_agents,),
            jnp.int32,
            [0] * self._env.num_agents,
            [self._env.time_limit] * self._env.num_agents,
            "step_count",
        )
        return self._env.observation_spec().replace(step_count=step_count)
