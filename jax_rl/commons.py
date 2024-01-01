from torch.utils.data import Dataset
import torch
import equinox as eqx
from jaxtyping import Array, Bool


class ReplayBuffer(eqx.Module):
    actions: Array
    rewards: Array
    dones: Bool[Array, "n_steps"]
    log_probs: Array
    states: Array

    def __init__(
        self,
        states: Array,
        actions: Array,
        rewards: Array,
        log_probs: Array,
        dones: Bool[Array, "n_steps"],
    ) -> None:
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.log_probs = log_probs
        self.dones = dones


class RLDataset(Dataset):
    def __init__(
        self,
        states: Array,
        actions: Array,
        rewards: Array,
        log_probs: Array,
        dones: Array,
    ) -> None:
        self.rewards = torch.tensor(rewards)
        self.actions = torch.tensor(actions)
        self.obs = torch.tensor(states)
        self.dones = torch.tensor(dones)
        self.log_probs = torch.tensor(log_probs)

    def __len__(self) -> int:
        return len(self.rewards)

    def __getitem__(self, idx) -> tuple:
        return (
            self.obs[idx].numpy(),
            self.actions[idx].numpy(),
            self.rewards[idx].numpy(),
            self.log_probs[idx].numpy(),
            self.dones[idx].numpy(),
        )
