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
from jumanji.types import Observation, TimeStep

from mava.types import State
from mava.wrappers.jumanji import MavaWrapper


class AgentIDWrapper(MavaWrapper):
    """Add onehot agent IDs to observation."""

    def __init__(self, env: MavaWrapper, has_global_state: bool = False):
        super().__init__(env)
        self._env: MavaWrapper
        self.has_global_state = has_global_state

    def _add_agent_ids(self, observation: chex.Array, num_agents: int) -> Observation:
        """Adds agent IDs to a (num_agents, num_features) observation."""
        chex.assert_rank(observation, 2)

        agent_ids = jnp.eye(num_agents)
        return jnp.concatenate([agent_ids, observation], axis=-1)

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep]:
        state, timestep = self._env.reset(key)
        new_view = self._add_agent_ids(timestep.observation.agents_view, self._env.num_agents)
        timestep.observation = timestep.observation._replace(agents_view=new_view)

        return state, timestep

    def step(
        self,
        state: State,
        action: chex.Array,
    ) -> Tuple[State, TimeStep]:
        state, timestep = self._env.step(state, action)
        new_view = self._add_agent_ids(timestep.observation.agents_view, self._env.num_agents)
        timestep.observation = timestep.observation._replace(agents_view=new_view)

        return state, timestep

    def observation_spec(self) -> specs.Spec:
        """Specification of the observation of the `RobotWarehouse` environment."""
        # assumes dim 0 is the number of agents
        # assumes dim 1 is the number of observation features

        obs_features = self._env.observation_spec().agents_view.shape[1]
        agents_view = specs.Array(
            (self._env.num_agents, obs_features + self._env.num_agents),
            jnp.int32,
            "agents_view",
        )
        return self._env.observation_spec().replace(
            agents_view=agents_view,
        )
