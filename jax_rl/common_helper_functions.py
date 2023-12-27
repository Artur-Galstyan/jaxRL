from typing import Any, NamedTuple, Optional

import equinox as eqx
import gymnasium as gym
import gymnax as gymx
import jax
import jax.numpy as jnp
import numpy as np
import torch
from beartype.typing import Callable
from gymnax.environments.environment import EnvParams
from jaxtyping import Array, Float, PRNGKeyArray
from torch.utils.data import Dataset


class RLDataset(Dataset):
    def __init__(self, states, actions, rewards, log_probs, dones) -> None:
        self.rewards = torch.tensor(rewards)
        self.actions = torch.tensor(actions)
        self.obs = torch.tensor(states)
        self.dones = torch.tensor(dones)
        self.log_probs = torch.tensor(log_probs)

    def __len__(self) -> int:
        return len(self.rewards)

    def __getitem__(
        self, idx
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.obs[idx],
            self.actions[idx],
            self.rewards[idx],
            self.log_probs[idx],
            self.dones[idx],
        )


def get_total_discounted_rewards(rewards: Float[Array, "n_steps"], gamma=0.99) -> Array:
    """Calculate the total discounted rewards for a given set of rewards.
    This is also known as the rewards-to-go.

    Args:
        rewards: The rewards to calculate the total discounted rewards for.
        gamma: The discount factor.
    Returns:
        The total discounted rewards.
    """

    def scan_fn(carry, current_reward):
        discounted_reward = carry + current_reward
        return discounted_reward * gamma, discounted_reward

    _, total_discounted_rewards = jax.lax.scan(scan_fn, 0.0, rewards[::-1])

    total_discounted_rewards = total_discounted_rewards[::-1].reshape(
        -1,
    )
    assert (
        total_discounted_rewards.shape == rewards.shape
    ), f"total_discounted_rewards.shape: {total_discounted_rewards.shape}, rewards.shape: {rewards.shape}"

    return total_discounted_rewards


def calculate_gae(
    rewards: Array,
    values: Array,
    dones: Array,
    gamma: float,
    lam: float,
) -> Array:
    def body_fun(
        carry: tuple[Array, Array], t: Array
    ) -> tuple[tuple[Array, Array], None]:
        advantages, gae_inner = carry
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        gae_inner = delta + gamma * lam * (1 - dones[t]) * gae_inner
        advantages = advantages.at[t].set(gae_inner)
        return (advantages, gae_inner), None

    values = jnp.append(values, values[0])
    advt = jnp.zeros_like(rewards)
    gae = jnp.array(0.0)
    t = len(rewards)

    (advt, _), _ = jax.lax.scan(body_fun, (advt, gae), jnp.arange(t - 1, -1, -1))
    return advt


def moving_average(data: Array, window_size: int):
    return jnp.convolve(data, jnp.ones(window_size), "valid") / window_size


def rollout_discrete(
    env: gymx.environments.environment.Environment | gym.Env,
    action_fn: Callable[[np.ndarray, dict], tuple[int, np.ndarray]],
    action_fn_kwargs: dict[Any, Any],
    key: PRNGKeyArray,
    env_params: Optional[EnvParams] = None,
    render: bool = False,
    max_steps: int = 200,
) -> tuple[RLDataset, dict]:
    if isinstance(env, gymx.environments.environment.Environment):
        return rollout_discrete_gymnax(
            env, env_params, action_fn, action_fn_kwargs, key, render, max_steps
        )
    elif isinstance(env, gym.Env):
        return rollout_discrete_gym(
            env, action_fn, action_fn_kwargs, key, render, max_steps
        )


def rollout_discrete_gymnax(
    env: gymx.environments.environment.Environment,
    env_params: EnvParams,
    action_fn: Callable[[np.ndarray, dict], tuple[int, np.ndarray]],
    action_fn_kwargs: dict[Any, Any],
    key: PRNGKeyArray,
    render: bool = False,
    max_steps: int = 200,
) -> tuple[RLDataset, dict]:
    key, rng_reset, rng_episode = jax.random.split(key, 3)

    obs, state = env.reset(rng_reset, env_params)

    @jax.jit
    def policy_step(state_input, tmp):
        obs, state, key = state_input
        key, step_key, action_key = jax.random.split(key, 3)
        action, logits = action_fn(obs, **action_fn_kwargs, key=action_key)
        log_prob = jnp.array(jax.nn.log_softmax(jnp.array(logits)))
        next_obs, next_state, reward, done, info = env.step(
            step_key, state, action, env_params
        )
        carry = [next_obs, next_state, key]

        return carry, [obs, action, reward, log_prob, next_obs, done, info]

    _, (obs, actions, rewards, log_probs, _, dones, info) = jax.lax.scan(
        policy_step,
        [obs, state, rng_episode],
        (),
        length=max_steps,
    )

    dataset = RLDataset(
        np.array(obs),
        np.array(actions),
        np.array(rewards),
        np.array(log_probs),
        np.array(dones),
    )

    return dataset, info


def rollout_discrete_gym(
    env: gym.Env,
    action_fn: Callable[[np.ndarray, dict], tuple[int, np.ndarray]],
    action_fn_kwargs: dict[Any, Any],
    key: PRNGKeyArray,
    render: bool = False,
    max_steps: Optional[int] = None,
) -> tuple[RLDataset, dict]:
    """Rollout a policy in a gym environment."""

    obs, info = env.reset()

    observations = []
    actions = []
    log_probs = []
    rewards = []
    dones = []
    info = None

    while True:
        key, subkey = jax.random.split(key)
        observations.append(obs)

        action, logits = action_fn(obs, **action_fn_kwargs, key=subkey)
        action = np.array(action)
        log_prob = np.array(jax.nn.log_softmax(np.array(logits)))
        log_probs.append(log_prob[action])
        obs, reward, terminated, truncated, info = env.step(action)
        if render:
            env.render()
        done = terminated or truncated
        actions.append(action)
        rewards.append(reward)
        dones.append(done)

        if max_steps is not None and len(observations) < max_steps:
            if done:
                obs, _ = env.reset()
        elif max_steps is not None and len(observations) >= max_steps:
            break
        else:
            if done:
                break

    dataset = RLDataset(
        np.array(observations),
        np.array(actions),
        np.array(rewards),
        np.array(log_probs),
        np.array(dones),
    )

    return dataset, info
