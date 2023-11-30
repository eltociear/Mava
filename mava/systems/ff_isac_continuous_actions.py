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

import copy
import time
from typing import Any, Callable, Dict, NamedTuple, Tuple

import distrax
import flashbax as fbx
import flax
import flax.linen as nn
import hydra
import jax
import jax.numpy as jnp
import jumanji
import optax
from chex import Array, Numeric, PRNGKey
from colorama import Fore, Style
from flashbax.buffers.flat_buffer import TransitionSample as BufferSample
from flashbax.buffers.trajectory_buffer import TrajectoryBuffer
from flashbax.buffers.trajectory_buffer import TrajectoryBufferState as BufferState
from jumanji.environments.routing.lbf.generator import RandomGenerator
from jumanji.wrappers import AutoResetWrapper
from omegaconf import DictConfig, OmegaConf
from rich.pretty import pprint

from JaxMARL.jaxmarl import make
from mava.evaluator import sac_evaluator_setup
from mava.logger import logger_setup
from mava.types import ActorApply, CriticApply, EvalState, LearnerState, Observation
from mava.wrappers.jaxmarl_wrapper import JaxMarlWrapper
from mava.wrappers.jumanji import AgentIDWrapper, LogWrapper, MultiAgentWrapper


def sample_action(mean: Array, log_std: Array, key: PRNGKey) -> Tuple[Array, Array]:
    std = jnp.exp(log_std)
    normal = distrax.Normal(mean, std)

    x_t = normal.sample(seed=key)
    y_t = jnp.tanh(x_t)
    # action = 2 * y_t  # actions [-2,2]
    # action = y_t(y_t * 0.5) + 0.5  # enforce actions between [0, 1]
    action = y_t
    log_prob = normal.log_prob(x_t)
    log_prob -= jnp.log((1 - y_t**2) + 1e-6)
    log_prob = jnp.sum(log_prob, axis=-1, keepdims=True)

    return action, log_prob


def select_actions_sac(
    apply_fn: Callable, params: nn.FrozenDict, obs: Array, key: PRNGKey
) -> Array:
    mean, log_std = apply_fn(params, obs)
    actions, _ = sample_action(mean, log_std, key)
    return actions


class Transition(NamedTuple):
    obs: Array
    action: Numeric
    reward: Array
    done: bool


class CriticParams(NamedTuple):
    """Parameters for a critic network since SAC uses two critics."""

    first: nn.FrozenDict
    second: nn.FrozenDict


class CriticAndTarget(NamedTuple):
    critics: CriticParams
    targets: CriticParams


class Params(NamedTuple):
    actor: nn.FrozenDict
    critic: CriticAndTarget
    log_alpha: Numeric


class OptStates(NamedTuple):
    actor: optax.OptState
    critic: optax.OptState
    alpha: optax.OptState


State = Tuple[LearnerState, BufferState[Transition]]


class Actor(nn.Module):
    """Actor Network."""

    action_dim: int
    # TODO: check this
    log_std_max: int = 2
    log_std_min: int = -5

    @nn.compact
    def __call__(self, observation: Observation) -> distrax.Categorical:
        """Forward pass."""
        x = observation.agents_view

        x = nn.tanh(nn.Dense(128)(x))
        x = nn.tanh(nn.Dense(128)(x))
        mean = nn.Dense(self.action_dim)(x)
        log_std = nn.tanh(nn.Dense(self.action_dim)(x))
        # From SpinUp / Denis Yarats
        log_std_range = self.log_std_max - self.log_std_min
        log_std = self.log_std_min + 0.5 * log_std_range * (log_std + 1)

        return mean, log_std


class Critic(nn.Module):
    """Actor Network."""

    action_dim: int

    @nn.compact
    def __call__(self, observation: Observation) -> distrax.Categorical:
        """Forward pass."""
        x = observation

        x = nn.relu(nn.Dense(128)(x))
        x = nn.relu(nn.Dense(128)(x))
        qs = nn.Dense(1)(x)

        # TODO: check this is correct
        # masked_qs = jnp.where(observation.action_mask, qs, -jnp.inf)
        return qs


# @partial(jax.pmap, axis_name="learner_devices", devices=learner_devices)
# def update(
#     sample: fbx.flat_buffer.TransitionSample[Transition],
#     actor_params: nn.FrozenDict,
#     critic_params: CriticAndTarget,
#     log_alpha: Numeric,
#     opt_states: Tuple[optax.OptState, optax.OptState, optax.OptState],
#     cfg: Dict,  # nested Dict[str: array]
#     key: PRNGKey,
# ):
#     # Reshape experience for the minibatch size
#     # (B, N, ...) -> (B // minibatch_size, minibatch_size, N, ...)
#     exp = jax.tree_map(
#         lambda x: jnp.reshape(
#             x,
#             (n_minibatches, cfg.system.minibatch_size, *x.shape[1:]),
#         ),
#         sample.experience,
#     )
#
#     def minibatch(i, carry):
#         exp, actor_params, critic_params, log_alpha, opt_states, _ = carry
#         actor_opt_state, critic_opt_state, alpha_opt_state = opt_states
#
#         obs = exp.first.obs[i]
#         next_obs = exp.second.obs[i]
#         act = exp.first.action[i]
#         rew = exp.first.reward[i]
#         done = exp.first.done
#
#         next_act_key, policy_loss_key, alpha_loss_key = jax.random.split(key, 3)
#
#         mean, log_std = vmapped_actor(actor_params, next_obs)
#         next_act, next_act_log_prob = sample_action(mean, log_std, next_act_key)
#
#         critic_input = jnp.concatenate([next_obs, next_act], axis=-1)
#
#         next_q1 = vmapped_critic(critic_params.targets.first, critic_input)
#         next_q2 = vmapped_critic(critic_params.targets.second, critic_input)
#         next_q = jnp.minimum(next_q1, next_q2).squeeze(axis=-1)
#
#         # (B, N)
#         # rew = jnp.expand_dims(rew, -1)
#         # done = jnp.expand_dims(done, -1)
#         target = rew + cfg.system.gamma * done * (
#             next_q - jnp.exp(log_alpha) * next_act_log_prob.squeeze(axis=-1)
#         )
#
#         c_loss, critic_grads = jax.value_and_grad(critic_loss)(
#             critic_params.critics, jnp.concatenate([obs, act], axis=-1), target
#         )
#         a_loss, actor_grads = jax.value_and_grad(policy_loss)(
#             actor_params, critic_params.critics, log_alpha, obs, policy_loss_key
#         )
#         alp_loss, alpha_grads = jax.value_and_grad(alpha_loss)(
#             log_alpha, actor_params, obs, alpha_loss_key
#         )
#
#         # todo: do a single pmean over a tuple of these?
#         # is that more performant?
#         actor_grads = jax.lax.pmean(actor_grads, "learner_devices")
#         critic_grads = jax.lax.pmean(critic_grads, "learner_devices")
#         alpha_grads = jax.lax.pmean(alpha_grads, "learner_devices")
#         a_loss, c_loss, alp_loss = jax.lax.pmean((a_loss, c_loss, alp_loss), "learner_devices")
#
#         # todo: join these updates into a single update
#         actor_updates, actor_opt_state = optim.update(actor_grads, actor_opt_state, actor_params)
#         critic_updates, critic_opt_state = optim.update(
#             critic_grads, critic_opt_state, critic_params.critics
#         )
#         alpha_updates, alpha_opt_state = optim.update(alpha_grads, alpha_opt_state, log_alpha)
#
#         actor_params = optax.apply_updates(actor_params, actor_updates)
#         new_critic_params = optax.apply_updates(critic_params.critics, critic_updates)
#         log_alpha = optax.apply_updates(log_alpha, alpha_updates)
#
#         new_target_params = optax.incremental_update(
#             critic_params.critics, critic_params.targets, cfg.system.tau
#         )
#
#         critic_params = CriticAndTarget(new_critic_params, new_target_params)
#
#         new_opt_states = (actor_opt_state, critic_opt_state, alpha_opt_state)
#         losses = (a_loss, c_loss, alp_loss)
#         return exp, actor_params, critic_params, log_alpha, new_opt_states, losses
#
#     init_val = (exp, actor_params, critic_params, log_alpha, opt_states, (0, 0, 0))
#     (
#         _,
#         actor_params,
#         critic_params,
#         log_alpha,
#         opt_states,
#         (a_loss, c_loss, alp_loss),
#     ) = jax.lax.fori_loop(0, n_minibatches, minibatch, init_val)
#
#     return (
#         actor_params,
#         critic_params,
#         log_alpha,
#         opt_states,
#         a_loss,
#         c_loss,
#         alp_loss,
#     )


def get_learner_fn(
    env: jumanji.Environment,
    # todo: just pass in actor and opt
    actor: Actor,
    critic: nn.Module,  # todo
    opt: optax.GradientTransformation,
    buffer: TrajectoryBuffer,
    config: Dict,
) -> Callable[[LearnerState, BufferState[Transition]], State]:
    n_rollouts = config["system"]["rollout_length"]
    target_entropy = -env.action_spec().shape[1]

    def critic_loss(critic_params: CriticParams, obs: Array, target: Array, acts: Array) -> Numeric:
        obs_act = jnp.concatenate([obs.agents_view, acts], axis=-1)
        q1 = critic.apply(critic_params.first, obs_act).squeeze(-1)
        q2 = critic.apply(critic_params.second, obs_act).squeeze(-1)
        return jnp.mean((target - q1) ** 2) + jnp.mean((target - q2) ** 2)

    def policy_loss(
        policy_params: nn.FrozenDict,
        critic_params: CriticParams,
        log_alpha: Numeric,
        obs: Observation,
        key: PRNGKey,
    ) -> Numeric:
        mean, log_std = actor.apply(policy_params, obs)
        acts, log_prob = sample_action(mean, log_std, key)

        critic_input = jnp.concatenate([obs.agents_view, acts], axis=-1)

        q1 = critic.apply(critic_params.first, critic_input).squeeze(axis=-1)
        q2 = critic.apply(critic_params.second, critic_input).squeeze(axis=-1)

        q = jnp.minimum(q1, q2)

        return jnp.mean(jnp.exp(log_alpha) * log_prob.squeeze(axis=-1) - q)

    def alpha_loss(
        log_alpha: Numeric, actor_params: nn.FrozenDict, obs: Array, key: PRNGKey
    ) -> Numeric:
        # todo: do this once! (double work here and in policy_loss)
        mean, log_std = actor.apply(actor_params, obs)
        _, log_prob = sample_action(mean, log_std, key)
        return -jnp.exp(log_alpha) * jnp.mean((log_prob + target_entropy))

    def update(
        learner_state: LearnerState[Params, OptStates], batch: BufferSample[Transition]
    ) -> Tuple[LearnerState, Dict[str, Array]]:
        obs = batch.experience.first.obs
        next_obs = batch.experience.second.obs
        acts = batch.experience.first.action
        rew = batch.experience.first.reward
        done = batch.experience.first.done

        key, next_act_key, policy_loss_key, alpha_loss_key = jax.random.split(learner_state.key, 4)
        learner_state = learner_state._replace(key=key)

        mean, log_std = actor.apply(learner_state.params.actor, next_obs)
        next_act, next_act_log_prob = sample_action(mean, log_std, next_act_key)

        critic_input = jnp.concatenate([next_obs.agents_view, next_act], axis=-1)

        next_q1 = critic.apply(learner_state.params.critic.targets.first, critic_input)
        next_q2 = critic.apply(learner_state.params.critic.targets.second, critic_input)
        next_q = jnp.minimum(next_q1, next_q2).squeeze(axis=-1)

        # (B, N)
        # rew = jnp.expand_dims(rew, -1)
        # done = jnp.expand_dims(done, -1)
        target = rew + config["system"]["gamma"] * (
            next_q - jnp.exp(learner_state.params.log_alpha) * next_act_log_prob.squeeze(axis=-1)
        )
        # (B, N)
        # rew = jnp.expand_dims(rew, -1)
        # done = jnp.expand_dims(done, -1)
        # target = rew[..., jnp.newaxis] + config["gamma"] * (1 - done[..., jnp.newaxis]) * (
        #     next_q - jnp.exp(learner_state.params.log_alpha) * next_act_log_prob
        # )

        c_loss, critic_grads = jax.value_and_grad(critic_loss)(
            learner_state.params.critic.critics, obs, target, acts
        )
        a_loss, actor_grads = jax.value_and_grad(policy_loss)(
            learner_state.params.actor,
            learner_state.params.critic.critics,
            learner_state.params.log_alpha,
            obs,
            policy_loss_key,
        )
        alp_loss, alpha_grads = jax.value_and_grad(alpha_loss)(
            learner_state.params.log_alpha, learner_state.params.actor, obs, alpha_loss_key
        )

        # todo: do a single pmean over a tuple of these?
        # is that more performant?
        # todo: add this back in when pmapping
        actor_grads = jax.lax.pmean(actor_grads, "batch")
        critic_grads = jax.lax.pmean(critic_grads, "batch")
        alpha_grads = jax.lax.pmean(alpha_grads, "batch")

        actor_grads = jax.lax.pmean(actor_grads, "device")
        critic_grads = jax.lax.pmean(critic_grads, "device")
        alpha_grads = jax.lax.pmean(alpha_grads, "device")
        # a_loss, c_loss, alp_loss = jax.lax.pmean((a_loss, c_loss, alp_loss), "batch")

        # todo: join these updates into a single update
        actor_updates, actor_opt_state = opt.update(
            actor_grads, learner_state.opt_states.actor, learner_state.params.actor
        )
        critic_updates, critic_opt_state = opt.update(
            critic_grads, learner_state.opt_states.critic, learner_state.params.critic.critics
        )
        alpha_updates, alpha_opt_state = opt.update(
            alpha_grads, learner_state.opt_states.alpha, learner_state.params.log_alpha
        )

        actor_params = optax.apply_updates(learner_state.params.actor, actor_updates)
        new_critic_params = optax.apply_updates(learner_state.params.critic.critics, critic_updates)
        log_alpha = optax.apply_updates(learner_state.params.log_alpha, alpha_updates)

        new_target_params = optax.incremental_update(
            learner_state.params.critic.critics,
            learner_state.params.critic.targets,
            config["system"]["tau"],
        )

        critic_params = CriticAndTarget(new_critic_params, new_target_params)

        new_opt_states = OptStates(
            actor=actor_opt_state, critic=critic_opt_state, alpha=alpha_opt_state
        )

        # Log mean_q
        critic_input = jnp.concatenate([obs.agents_view, acts], axis=-1)
        q1 = critic.apply(new_critic_params.first, critic_input)
        q2 = critic.apply(new_critic_params.second, critic_input)

        q = jnp.minimum(q1, q2)
        mean_q = jnp.mean(q)

        losses = {
            "actor_loss": a_loss,
            "critic_loss": c_loss,
            "alpha_loss": alp_loss,
            "alpha": jnp.exp(log_alpha),
            "mean_q": mean_q,
        }
        losses = jax.lax.pmean(losses, "device")
        learner_state = LearnerState(
            params=Params(actor=actor_params, critic=critic_params, log_alpha=log_alpha),
            opt_states=new_opt_states,
            key=key,
            env_state=learner_state.env_state,
            timestep=learner_state.timestep,
        )
        return learner_state, losses

    def act(carry: State, _: Any) -> State:
        learner_state, buffer_state = carry
        actor_params = learner_state.params.actor

        # SELECT ACTION
        action_key = jax.random.split(learner_state.key, config["arch"]["num_envs"])
        action = jax.vmap(select_actions_sac, in_axes=(None, None, 0, 0))(
            actor.apply, actor_params, learner_state.timestep.observation, action_key
        )

        # STEP ENVIRONMENT
        env_state, timestep = jax.vmap(env.step)(learner_state.env_state, action)

        # LOG EPISODE METRICS
        done = jax.tree_util.tree_map(
            lambda x: jnp.repeat(x, config["system"]["num_agents"]).reshape(
                config["arch"]["num_envs"], -1
            ),
            timestep.last(),
        )
        reward = timestep.reward

        info = {
            "episode_return": env_state.episode_return_info,
            "episode_length": env_state.episode_length_info,
        }

        # learner_state.timestep is the obs and timestep is next obs
        obs = learner_state.timestep.observation  # todo - save whole obs?
        transition = Transition(obs=obs, action=action, reward=reward, done=done)
        learner_state = learner_state._replace(
            env_state=env_state, timestep=timestep, key=action_key[0]
        )
        # todo: check if the donate_argnums is preserved here
        buffer_state = buffer.add(buffer_state, transition)
        return (learner_state, buffer_state), info

    def _act_and_log(carry: State, _: Any) -> State:
        learner_state, buffer_state = carry

        key, sample_key = jax.random.split(learner_state.key)
        learner_state = learner_state._replace(key=key)

        (learner_state, buffer_state), metrics = jax.lax.scan(
            act, (learner_state, buffer_state), None, n_rollouts
        )

        # todo: clean this up a bit :)
        def learn(learner_state):
            batches = buffer.sample(buffer_state, sample_key)
            minibatch_size = int(
                config["system"]["batch_size"] / config["system"]["num_minibatches"]
            )
            batches = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, (-1, minibatch_size, *x.shape[1:])), batches
            )
            # todo treemap -> reshape(num_minibatches, ...)
            # todo: get metrics/losses from here

            learner_state, loss_info = jax.lax.scan(update, learner_state, batches)
            return learner_state, loss_info

        def noop(learner_state):
            return learner_state, {
                # todo: make sure its num_minibatches and not num_envs
                "actor_loss": jnp.zeros(config["system"]["num_minibatches"]),
                "critic_loss": jnp.zeros(config["system"]["num_minibatches"]),
                "alpha_loss": jnp.zeros(config["system"]["num_minibatches"]),
                "alpha": jnp.zeros(config["system"]["num_minibatches"]),
                "mean_q": jnp.zeros(config["system"]["num_minibatches"]),
            }

        learner_state, loss_info = jax.lax.cond(
            buffer.can_sample(buffer_state), learn, noop, learner_state
        )

        return (learner_state, buffer_state), (metrics, loss_info)

    def act_and_learn(learner_state: LearnerState, buffer_state: BufferState[Transition]) -> State:
        batched_update_step = jax.vmap(_act_and_log, in_axes=(0, None), axis_name="batch")
        (learner_state, buffer_state), (metrics, loss_info) = jax.lax.scan(
            batched_update_step,
            (learner_state, buffer_state),
            None,
            (config["system"]["num_updates"] // config["arch"]["num_evaluation"]),
        )
        return (learner_state, buffer_state), (metrics, loss_info)

    return act_and_learn


def main(_config) -> None:
    config = copy.deepcopy(_config)
    log = logger_setup(config, system="sac")

    key = jax.random.PRNGKey(config["system"]["seed"])

    env = make("ant_4x2")
    env = JaxMarlWrapper(env)

    # Add agent id to observation.
    # if config["system"]["add_agent_id"]:
    #     env = AgentIDWrapper(env)
    env = AutoResetWrapper(env)
    env = LogWrapper(env)

    config["system"]["num_agents"] = env.action_spec().shape[0]

    actor = Actor(env.action_spec().shape[1])
    critic = Critic(env.action_spec().shape[1])  # todo: better critic
    opt = optax.adam(config["system"]["lr"])
    buffer = fbx.make_flat_buffer(
        config["system"]["rb_size"],
        0,
        config["system"]["batch_size"],
        add_batch_size=config["arch"]["num_envs"],
    )

    dummy_act = env.action_spec().generate_value()
    dummy_obs = env.observation_spec().generate_value()
    dummy_obs = Observation(
        agents_view=dummy_obs.agents_view,
        action_mask=dummy_obs.action_mask,
        step_count=dummy_obs.step_count,
    )
    dummy_transition = Transition(
        obs=dummy_obs,
        action=dummy_act,
        reward=jnp.zeros(env.num_agents),
        done=jnp.zeros(env.num_agents, dtype=bool),
    )
    critic_input = jnp.concatenate([dummy_obs.agents_view, dummy_act], axis=-1)
    key, critic_1_key, critic_2_key, critic_3_key, critic_4_key = jax.random.split(key, 5)
    params = Params(
        actor=actor.init(key, dummy_obs),
        # todo better names, this is confusing
        # critics -> online
        critic=CriticAndTarget(
            critics=CriticParams(
                first=critic.init(critic_1_key, critic_input),
                second=critic.init(critic_2_key, critic_input),
            ),
            targets=CriticParams(
                first=critic.init(critic_3_key, critic_input),
                second=critic.init(critic_4_key, critic_input),
            ),
        ),
        # todo: separate log alphas
        log_alpha=jnp.asarray(0.0),
    )
    opt_states = OptStates(
        # todo: allow for different optimizers and different learning rates.
        actor=opt.init(params.actor),
        critic=opt.init(params.critic.critics),
        alpha=opt.init(params.log_alpha),
    )
    buffer_state = buffer.init(dummy_transition)

    reset_keys = jax.random.split(key, num=config["arch"]["num_envs"])  # todo: num_envs
    state, timestep = jax.vmap(env.reset)(reset_keys)
    learner_state = LearnerState(
        params=params,
        opt_states=opt_states,
        key=key,
        env_state=state,
        timestep=timestep,
    )

    config["arch"]["devices"] = len(jax.devices())
    steps_per_rollout = (
        config["arch"]["devices"]
        * (config["system"]["num_updates"] // config["arch"]["num_evaluation"])
        * config["system"]["rollout_length"]
        * config["arch"]["num_envs"]
    )
    pprint(config)

    # Get learner and evaluator functions.
    learner_fn = get_learner_fn(env, actor, critic, opt, buffer, config)
    learner_fn = jax.pmap(learner_fn, axis_name="device")
    evaluator, _ = sac_evaluator_setup(
        eval_env=env,
        rng_e=key,
        network=actor,
        params=params.actor,
        config=config,
    )

    # Duplicate learner for update_batch_size.
    broadcast = lambda x: jnp.broadcast_to(x, (config["system"]["update_batch_size"],) + x.shape)
    learner_state = jax.tree_map(broadcast, learner_state)
    buffer_state = jax.tree_map(broadcast, buffer_state)

    # Duplicate learner and buffer states across devices.
    learner_state = flax.jax_utils.replicate(learner_state, devices=jax.devices())
    buffer_state = flax.jax_utils.replicate(buffer_state, devices=jax.devices())
    for eval_i in range(config["arch"]["num_evaluation"]):
        start_time = time.time()
        key, eval_key = jax.random.split(key)

        learner_output = learner_fn(learner_state, buffer_state)
        jax.block_until_ready(learner_output)

        (learner_state, buffer_state), metrics = learner_output
        elapsed_time = time.time() - start_time
        metrics[0]["steps_per_second"] = steps_per_rollout / elapsed_time
        log(
            metrics,
            t_env=(eval_i + 1) * steps_per_rollout,
            trainer_metric=True,
        )

        start_time = time.time()
        trained_actor = jax.tree_util.tree_map(
            lambda x: x[:, 0, ...], learner_state.params.actor  # Select only actor params
        )
        eval_key, *eval_rngs = jax.random.split(eval_key, config["arch"]["devices"] + 1)
        eval_rngs = jnp.stack(eval_rngs).reshape(config["arch"]["devices"], -1)
        evaluator_output = evaluator(trained_actor, eval_rngs)
        elapsed_time = time.time() - start_time
        evaluator_output.episodes_info["steps_per_second"] = steps_per_rollout / elapsed_time
        log(evaluator_output, t_env=(eval_i + 1) * steps_per_rollout, trainer_metric=False)


@hydra.main(config_path="../configs", config_name="default_ff_isac.yaml", version_base="1.2")
def hydra_entry_point(cfg: DictConfig) -> None:
    """Experiment entry point."""
    # Convert config to python dict.
    cfg: Dict = OmegaConf.to_container(cfg, resolve=True)

    # Run experiment.
    main(cfg)

    print(f"{Fore.CYAN}{Style.BRIGHT}ISAC experiment completed{Style.RESET_ALL}")


if __name__ == "__main__":
    hydra_entry_point()
