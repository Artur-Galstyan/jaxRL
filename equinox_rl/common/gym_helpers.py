from typing import Callable
import jax.numpy as jnp
from jaxtyping import Array
import gymnasium as gym


def rollout_discrete(
    env: gym.Env, action_fn: Callable, action_fn_kwargs: dict
) -> tuple[Array, Array, Array, Array]:
    obs, _ = env.reset()

    observations = []
    actions = []
    rewards = []
    dones = []

    while True:
        observations.append(obs)

        action = action_fn(obs, **action_fn_kwargs)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        actions.append(action)
        rewards.append(reward)
        dones.append(done)

        if done:
            break

    return (
        jnp.array(observations),
        jnp.array(actions),
        jnp.array(rewards),
        jnp.array(dones),
    )
