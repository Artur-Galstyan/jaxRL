from typing import Optional
from torch.utils.data import Dataset
import torch
import equinox as eqx
from jaxtyping import Array, Bool
import jax.numpy as jnp
import jax


class ReplayBuffer(eqx.Module):
    actions: Array
    rewards: Array
    dones: Bool[Array, "n_steps"]
    log_probs: Array
    states: Array
    values: Optional[Array]

    def __init__(
        self,
        states: Array,
        actions: Array,
        rewards: Array,
        log_probs: Array,
        dones: Bool[Array, "n_steps"],
        values: Optional[Array] = None,
    ) -> None:
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.log_probs = log_probs
        self.dones = dones
        self.values = values


class RLDataset(Dataset):
    def __init__(
        self,
        states: Array,
        actions: Array,
        rewards: Array,
        log_probs: Array,
        dones: Array,
        values: Optional[Array] = None,
    ) -> None:
        self.rewards = torch.tensor(rewards)
        self.actions = torch.tensor(actions)
        self.obs = torch.tensor(states)
        self.dones = torch.tensor(dones)
        self.log_probs = torch.tensor(log_probs)
        self.values = torch.tensor(values) if values is not None else None

    def __len__(self) -> int:
        return len(self.rewards)

    def __getitem__(self, idx) -> tuple:
        return (
            self.obs[idx].numpy(),
            self.actions[idx].numpy(),
            self.rewards[idx].numpy(),
            self.log_probs[idx].numpy(),
            self.dones[idx].numpy(),
            self.values[idx].numpy() if self.values is not None else None,
        )


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
