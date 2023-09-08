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
from collections import namedtuple
from typing import Any, NamedTuple, Tuple

import chex
import jax.numpy as jnp
from jumanji import specs
from jumanji.env import Environment
from jumanji.types import TimeStep

from mava.types import State
from mava.wrappers.base import MavaWrapper

# todo: jumanji does not allow for nested observations...
# could put it in timestep.extras, but then no spec - get_global_state_spec() and get_global_state()
# could create namedtuple at runtime, but yuck (maybe its not so bad cause duck typing?)
# could create an observation per env, but then not general
# class ObservationGlobalState(NamedTuple):
#     """The observation that the agent sees.
#
#     observation: the observation of the agents.
#     global_state: the global state of the environment, which can the
#         concatenation of the agents' observations or a defined global state.
#     """
#
#     observation: Any
#     global_state: chex.Array


class GlobalStateWrapper(MavaWrapper, ABC):
    """Multi-agent wrapper for the Robotic Warehouse environment.

    The wrapper includes a global environment state to be used by the centralised critic.
    Note here that since robotic warehouse does not have a global state, we create one
    by concatenating the observations of all agents.
    """

    def __init__(self, env: Environment):
        super().__init__(env)

        # creating observation global state at runtime...not sure I like this?
        # creates a namedtuple with the same fields as the observation along with global_state
        observation = env.observation_spec().generate_value()
        self.ObservationGlobalState = namedtuple(
            "ObservationGlobalState", observation._fields + ("global_state",)
        )

    @abstractmethod
    def get_global_state(self, state: State, timestep: TimeStep) -> chex.Array:
        """Get the global state of the environment."""
        pass

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep]:
        """Reset the environment. Updates the step count."""
        state, timestep = self._env.reset(key)

        global_state = self.get_global_state(state, timestep)
        timestep.observation = self.ObservationGlobalState(
            observation=timestep.observation, global_state=global_state
        )

        return state, timestep

    def step(self, state: State, action: chex.Array) -> Tuple[State, TimeStep]:
        """Step the environment. Updates the step count."""
        state, timestep = self._env.step(state, action)

        global_state = self.get_global_state(state, timestep)
        timestep.observation = self.ObservationGlobalState(
            observation=timestep.observation, global_state=global_state
        )

        return state, timestep

    def observation_spec(self) -> specs.Spec:
        # todo: agents_view is not general :(
        # Don't want to have to make a custom one of these for each env
        print(self._env.observation_spec())
        print(self._env.observation_spec().agents_view)
        obs_features = self._env.observation_spec().agents_view.shape[1]
        global_state = specs.Array(
            (self._env.num_agents, self._env.num_agents * obs_features),
            jnp.int32,
            "global_state",
        )

        return specs.Spec(
            self.ObservationGlobalState,
            "ObservationSpec",
            observation=self._env.observation_spec(),
            global_state=global_state,
        )
