import gymnasium as gym
import jax
import numpy as np
import torch
from beartype.typing import Callable
from jaxtyping import PRNGKeyArray
from torch.utils.data.dataset import Dataset


def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size), "valid") / window_size


class RLDataset(Dataset):
    def __init__(self, states, actions, rewards, dones) -> None:
        self.rewards = torch.tensor(rewards)
        self.actions = torch.tensor(actions)
        self.obs = torch.tensor(states)
        self.dones = torch.tensor(dones)

    def __len__(self) -> int:
        return len(self.rewards)

    def __getitem__(
        self, idx
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.obs[idx],
            self.actions[idx],
            self.rewards[idx],
            self.dones[idx],
        )


def rollout_discrete(
    env: gym.Env,
    action_fn: Callable,
    action_fn_kwargs: dict,
    key: PRNGKeyArray,
    render: bool = False,
) -> tuple[RLDataset, dict]:
    obs, _ = env.reset()

    observations = []
    actions = []
    rewards = []
    dones = []
    info = None

    while True:
        key, subkey = jax.random.split(key)
        observations.append(obs)

        action = np.array(action_fn(obs, **action_fn_kwargs, key=subkey))
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        actions.append(action)
        rewards.append(reward)
        dones.append(done)

        if done:
            break

    dataset = RLDataset(
        np.array(observations), np.array(actions), np.array(rewards), np.array(dones)
    )

    return dataset, info
