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

from abc import ABC

from jumanji.wrappers import Wrapper


# Would also be nice to enfoce the observation names here if possible
class MavaWrapper(Wrapper, ABC):
    """Simple wrapper around the Jumanji wrapper which enforces
    the `num_agents` and `time_limit` properties.
    """

    # Again a little weird, this will almost always be true except for the
    # first time the env is wrapped. But it sorts out the typing.
    # def __init__(self, env: "MavaWrapper"):
    #     super().__init__(env)
    #     self._env: "MavaWrapper"

    # This is a bit funky:
    # I want users to have to specify the number of agents and time limit for all envs.
    # however, I don't want all wrappers to have to implement these properties.
    # so we can't make this an abstractmethod as some wrappers won't need to implement it.
    @property
    def num_agents(self) -> int:
        """Number of agents in the environment."""
        return self._env.num_agents  # type: ignore

    @property
    def time_limit(self) -> int:
        """Time limit for the environment."""
        return self._env.num_agents  # type: ignore
