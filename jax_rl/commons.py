from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool


class ReplayBuffer(eqx.Module):
    actions: Array
    rewards: Array
    dones: Bool[Array, " n_steps"]
    log_probs: Array
    states: Array
    values: Optional[Array]

    def __init__(
        self,
        states: Array,
        actions: Array,
        rewards: Array,
        log_probs: Array,
        dones: Bool[Array, " n_steps"],
        values: Optional[Array] = None,
    ) -> None:
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.log_probs = log_probs
        self.dones = dones
        self.values = values


def calculate_gae(
    rewards: Array,
    values: Array,
    dones: Array,
    gamma: float,
    lambda_: float,
) -> Array:
    def body_fun(
        carry: tuple[Array, Array], t: Array
    ) -> tuple[tuple[Array, Array], None]:
        advantages, gae_inner = carry
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        gae_inner = delta + gamma * lambda_ * (1 - dones[t]) * gae_inner
        advantages = advantages.at[t].set(gae_inner)
        return (advantages, gae_inner), None

    values = jnp.append(values, values[0])
    advt = jnp.zeros_like(rewards)
    gae = jnp.array(0.0)
    t = len(rewards)

    (advt, _), _ = jax.lax.scan(body_fun, (advt, gae), jnp.arange(t - 1, -1, -1))
    return advt
