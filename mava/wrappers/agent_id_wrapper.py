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

from typing import Tuple, Union

import chex
import jax.numpy as jnp
from jumanji import specs
from jumanji.types import Observation, TimeStep

from mava.types import State
from mava.wrappers.global_state_wrapper import ObservationGlobalState
from mava.wrappers.jumanji import MavaWrapper


class AgentIDWrapper(MavaWrapper):
    """Add onehot agent IDs to observation."""

    def __init__(self, env: MavaWrapper, has_global_state: bool = False):
        super().__init__(env)
        self._env: MavaWrapper
        self.has_global_state = has_global_state

    def _add_agent_ids(
        self, timestep: TimeStep, num_agents: int
    ) -> Union[Observation, ObservationGlobalState]:
        agent_ids = jnp.eye(num_agents)

        # todo: we can't have this if statement here. Wrappers should not have to know
        # about other wrappers.
        # options: make a 2 wrappers one with only IDs and one with IDs+global state
        # not sure of any other options?
        if self.has_global_state:
            # todo: again assuming agents_view is in obs
            new_agents_view = jnp.concatenate(
                [agent_ids, timestep.observation.observation.agents_view], axis=-1
            )
            # Add the agent IDs to the global state
            new_global_state = jnp.concatenate(
                [agent_ids, timestep.observation.global_state], axis=-1
            )

            return ObservationGlobalState(
                observation=new_agents_view,
                global_state=new_global_state,
            )

        else:
            # todo: again assuming agents_view is in obs
            new_agents_view = jnp.concatenate(
                [agent_ids, timestep.observation.agents_view], axis=-1
            )
            return timestep.observation._replace(agents_view=new_agents_view)

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep]:
        """Reset the environment."""
        state, timestep = self._env.reset(key)
        timestep.observation = self._add_agent_ids(timestep, self._env.num_agents)

        return state, timestep

    def step(
        self,
        state: State,
        action: chex.Array,
    ) -> Tuple[State, TimeStep]:
        """Step the environment."""
        state, timestep = self._env.step(state, action)
        timestep.observation = self._add_agent_ids(timestep, self._env.num_agents)

        return state, timestep

    def observation_spec(self) -> specs.Spec[Observation]:
        """Specification of the observation of the `RobotWarehouse` environment."""
        # assumes dim 0 is the number of agents
        # assumes dim 1 is the number of observation features

        if self.has_global_state:
            obs_features = self._env.observation_spec().observation.agents_view.shape[1]
            agents_view = specs.Array(
                (self._env.num_agents, obs_features + self._env.num_agents),
                jnp.int32,
                "agents_view",
            )
            global_state = specs.Array(
                (self._env.num_agents, obs_features * self._env.num_agents + self._env.num_agents),
                jnp.int32,
                "global_state",
            )

            return self._env.observation_spec().replace(
                agents_view=agents_view,
                global_state=global_state,
            )
        else:
            obs_features = self._env.observation_spec().agents_view.shape[1]
            agents_view = specs.Array(
                (self._env.num_agents, obs_features + self._env.num_agents),
                jnp.int32,
                "agents_view",
            )
            return self._env.observation_spec().replace(
                agents_view=agents_view,
            )
