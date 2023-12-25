from typing import Optional
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.obs[idx],
            self.actions[idx],
            self.rewards[idx],
            self.log_probs[idx],
            self.dones[idx],
        )


def rollout_discrete(
    env: gym.Env,
    action_fn: Callable[[np.ndarray, dict], tuple[int, np.ndarray]],
    action_fn_kwargs: dict,
    key: PRNGKeyArray,
    render: bool = False,
    max_steps: Optional[int] = None,
) -> tuple[RLDataset, dict]:
    obs, _ = env.reset()

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
