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
from jumanji.types import TimeStep

from mava.types import LogEnvState
from mava.wrappers.base import MavaWrapper


class LogWrapper(MavaWrapper):
    """Log the episode returns and lengths."""

    def reset(self, key: chex.PRNGKey) -> Tuple[LogEnvState, TimeStep]:
        """Reset the environment."""
        state, timestep = self._env.reset(key)
        state = LogEnvState(state, jnp.float32(0.0), 0, jnp.float32(0.0), 0)
        return state, timestep

    def step(
        self,
        state: LogEnvState,
        action: chex.Array,
    ) -> Tuple[LogEnvState, TimeStep]:
        """Step the environment."""
        env_state, timestep = self._env.step(state.env_state, action)

        done = timestep.last()
        not_done = 1 - done

        new_episode_return = state.episode_returns + jnp.mean(timestep.reward)
        new_episode_length = state.episode_lengths + 1
        episode_return_info = state.episode_return_info * not_done + new_episode_return * done
        episode_length_info = state.episode_length_info * not_done + new_episode_length * done

        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * not_done,
            episode_lengths=new_episode_length * not_done,
            episode_return_info=episode_return_info,
            episode_length_info=episode_length_info,
        )
        return state, timestep
