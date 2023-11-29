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

from typing import TYPE_CHECKING, Any, Callable, Dict, Generic, Optional, Tuple, TypeVar

import chex
from distrax import Distribution
from flax.core.frozen_dict import FrozenDict
from jumanji.types import TimeStep
from optax._src.base import OptState
from typing_extensions import NamedTuple, TypeAlias

if TYPE_CHECKING:  # https://github.com/python/mypy/issues/6239
    from dataclasses import dataclass
else:
    from flax.struct import dataclass


Action: TypeAlias = chex.Array
Value: TypeAlias = chex.Array
Done: TypeAlias = chex.Array
HiddenState: TypeAlias = chex.Array
# Can't know the exact type of State.
State: TypeAlias = Any


class Observation(NamedTuple):
    """The observation that the agent sees.
    agents_view: the agents' view of other agents and shelves within their
        sensor range. The number of features in the observation array
        depends on the sensor range of the agent.
    action_mask: boolean array specifying, for each agent, which action is legal.
    step_count: the number of steps elapsed since the beginning of the episode.
    """

    agents_view: chex.Array
    action_mask: chex.Array
    step_count: chex.Numeric


class ObservationGlobalState(NamedTuple):
    """The observation seen by agents in centralized systems.
    Extends `Observation` by adding a `global_state` attribute for centralized training.
    global_state: The global state of the environment, often a concatenation of agents' views.
    """

    agents_view: chex.Array  # (num_agents, num_obs_features)
    action_mask: chex.Array  # (num_agents, num_actions)
    global_state: chex.Array  # (num_agents, num_agents * num_obs_features, )
    step_count: chex.Array  # (num_agents, )


@dataclass
class LogEnvState:
    """State of the `LogWrapper`."""

    env_state: State
    episode_returns: chex.Numeric
    episode_lengths: chex.Numeric
    # Information about the episode return and length for logging purposes.
    episode_return_info: chex.Numeric
    episode_length_info: chex.Numeric


RnnObservation: TypeAlias = Tuple[Observation, Done]
RnnGlobalObservation: TypeAlias = Tuple[ObservationGlobalState, Done]


class PPOTransition(NamedTuple):
    """Transition tuple for PPO."""

    done: Done
    action: Action
    value: Value
    reward: chex.Array
    log_prob: chex.Array
    obs: chex.Array
    info: Dict


class Params(NamedTuple):
    """Parameters of an actor critic network."""

    actor_params: FrozenDict
    critic_params: FrozenDict


class OptStates(NamedTuple):
    """OptStates of actor critic learner."""

    actor_opt_state: OptState
    critic_opt_state: OptState


class HiddenStates(NamedTuple):
    """Hidden states for an actor critic learner."""

    policy_hidden_state: HiddenState
    critic_hidden_state: HiddenState


TParams = TypeVar("TParams")
TOptStates = TypeVar("TOptStates")


class LearnerState(NamedTuple, Generic[TParams, TOptStates]):
    """State of the learner."""

    params: TParams
    opt_states: TOptStates
    key: chex.PRNGKey
    env_state: LogEnvState
    timestep: TimeStep


class RNNLearnerState(NamedTuple):
    """State of the `Learner` for recurrent architectures."""

    params: Params
    opt_states: OptStates
    key: chex.PRNGKey
    env_state: LogEnvState
    timestep: TimeStep
    dones: Done
    hstates: HiddenStates


class EvalState(NamedTuple):
    """State of the evaluator."""

    key: chex.PRNGKey
    env_state: State
    timestep: TimeStep
    step_count_: chex.Numeric
    return_: chex.Numeric


class RNNEvalState(NamedTuple):
    """State of the evaluator for recurrent architectures."""

    key: chex.PRNGKey
    env_state: State
    timestep: TimeStep
    dones: chex.Array
    hstate: HiddenState
    step_count_: chex.Numeric
    return_: chex.Numeric


MavaState = TypeVar("MavaState", LearnerState, RNNLearnerState, EvalState, RNNEvalState)


class ExperimentOutput(NamedTuple, Generic[MavaState]):
    """Experiment output."""

    episodes_info: Dict[str, chex.Array]
    learner_state: MavaState
    # these aren't common between value and policy methods
    # should likely just be a dict of metrics
    total_loss: Optional[chex.Array] = None
    value_loss: Optional[chex.Array] = None
    loss_actor: Optional[chex.Array] = None
    entropy: Optional[chex.Array] = None


LearnerFn = Callable[[MavaState], ExperimentOutput[MavaState]]
EvalFn = Callable[[FrozenDict, chex.PRNGKey], ExperimentOutput[MavaState]]

ActorApply = Callable[[FrozenDict, Observation], Distribution]
CriticApply = Callable[[FrozenDict, Observation], Value]
RecActorApply = Callable[
    [FrozenDict, HiddenState, RnnObservation], Tuple[HiddenState, Distribution]
]
RecCriticApply = Callable[[FrozenDict, HiddenState, RnnObservation], Tuple[HiddenState, Value]]
