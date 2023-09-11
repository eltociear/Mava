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

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import chex
import jax.numpy as jnp
from jumanji import specs
from jumanji.types import TimeStep

from mava.types import State
from mava.wrappers.agent_id_wrapper import AgentIDWrapper
from mava.wrappers.base import MavaWrapper


class GlobalStateWrapper(MavaWrapper, ABC):
    """Multi-agent wrapper for the Robotic Warehouse environment.

    The wrapper includes a global environment state to be used by the centralised critic.
    Note here that since robotic warehouse does not have a global state, we create one
    by concatenating the observations of all agents.
    """

    GLOBAL_STATE = "global_state"

    # just for type checking
    def __init__(self, env: MavaWrapper):
        super().__init__(env)
        self._env: MavaWrapper

    @abstractmethod
    def _get_global_state(self, state: State, timestep: TimeStep) -> chex.Array:
        """Get the global state of the environment."""
        pass

    @abstractmethod
    def global_state_spec(self) -> specs.Array:
        """Get the global state spec."""
        pass


class DefaultGlobalStateWrapper(GlobalStateWrapper):
    """The default global state wrapper, this will work for observations that are
    of shape (num_agents, num_features) and live in `timestep.observation.agents_view`.

    By default the global state is the concatenation of the agents' observations.

    Places the global state in `timestep.observation.extras[global_state]`.
    """

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep]:
        state, timestep = super().reset(key)
        extras = timestep.observation.extras or {}  # create if not exists
        extras[self.GLOBAL_STATE] = self._get_global_state(state, timestep)
        timestep.observation.extras = extras

        return state, timestep

    def step(self, state: State, action: chex.Array) -> Tuple[State, TimeStep]:
        state, timestep = super().step(state, action)
        extras = timestep.extras or {}  # create if not exists
        extras["global_state"] = self._get_global_state(state, timestep)
        timestep.observation.extras = extras

        return state, timestep

    def _get_global_state(self, state: State, timestep: TimeStep) -> chex.Array:
        """Get the global state of the environment."""
        global_state = jnp.concatenate(timestep.observation.agents_view, axis=0)
        return jnp.tile(global_state, (self._env.num_agents, 1))

    def global_state_spec(self) -> specs.Array:
        obs_features = self._env.observation_spec().agents_view.shape[1]
        return specs.Array(
            (self._env.num_agent, self._env.num_agents * obs_features),
            jnp.int32,
            "global_state",
        )


class GlobalStateWithAgentIDWrapper(GlobalStateWrapper):
    def __init__(
        self,
        env: MavaWrapper,
        global_state_wrapper: Optional[GlobalStateWrapper] = None,
        agent_id_wrapper: Optional[AgentIDWrapper] = None,
    ):
        """Adds global state observation and agent IDs to the environment.
        Global state is added in timestep.extras[global_state] and agent IDs are added
        are added to both the global state and timestep.observation.agents_view.

        Any type of global state can be added by passing a custom global_state_wrapper, by default
        this is `DefaultGlobalStateWrapper` which concatenates the agents' observations. Similarly,
        any type of agent ID wrapper can be passed, by default this is `AgentIDWrapper` which adds
        agent IDs to the `observation.agents_view`.

        Args:
            env: the environment to wrap.
            global_state_wrapper: the global state wrapper to use.
            agent_id_wrapper: the agent ID wrapper to use.
        """
        super().__init__(env)
        self._global_state_wrapper = global_state_wrapper or DefaultGlobalStateWrapper(env)
        self._agent_id_wrapper = agent_id_wrapper or AgentIDWrapper(env)

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep]:
        state, timestep = self._env.reset(key)

        # get global state - must be before adding agent IDs to observation
        global_state = self._global_state_wrapper._get_global_state(state, timestep)
        # add agent IDs to global state
        global_state = self._agent_id_wrapper._add_agent_ids(global_state, self._env.num_agents)

        # put it in timestep.extras
        extras = timestep.extras or {}  # create if not exists
        extras[self.GLOBAL_STATE] = global_state
        timestep.extras = global_state

        # add agent IDs to observation.agents_view
        agents_view = timestep.observation.agents_view
        new_view = self._agent_id_wrapper._add_agent_ids(agents_view, self._env.num_agents)
        timestep.observation = timestep.observation._replace(agents_view=new_view)

        return state, timestep

    def step(self, state: State, action: chex.Array) -> Tuple[State, TimeStep]:
        state, timestep = self._env.step(state, action)

        # get global state - must be before adding agent IDs to observation
        global_state = self._global_state_wrapper._get_global_state(state, timestep)
        # add agent IDs to global state
        global_state = self._agent_id_wrapper._add_agent_ids(global_state, self._env.num_agents)

        # put it in timestep.extras
        extras = timestep.extras or {}  # create if not exists
        extras[self.GLOBAL_STATE] = global_state
        timestep.extras = global_state

        # add agent IDs to observation.agents_view
        agents_view = timestep.observation.agents_view
        new_view = self._agent_id_wrapper._add_agent_ids(agents_view, self._env.num_agents)
        timestep.observation = timestep.observation._replace(agents_view=new_view)

        return state, timestep

    def global_state_spec(self) -> specs.Array:
        global_state_spec = self._global_state_wrapper.global_state_spec()
        global_state_shape = global_state_spec.shape
        new_shape = (global_state_shape[0], global_state_shape[1] + self._env.num_agents)

        return specs.Array(new_shape, jnp.int32, "global_state")
