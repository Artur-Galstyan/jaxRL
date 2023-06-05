import pathlib
from typing import Tuple

import common.rl_helpers as rlh
import equinox as eqx
import gymnasium as gym
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float32, PRNGKeyArray, PyTree
import numpy as np
import optax
import optuna
import tensorflow_probability.substrates.jax as tfp
import torch
from common import eqx_helpers
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm import tqdm


class RLDataset(Dataset):
    def __init__(self, states, actions, rewards, dones):
        self.rewards = rewards
        self.actions = actions
        self.obs = states
        self.dones = dones

    def __len__(self):
        return len(self.rewards)

    def __getitem__(self, index):
        reward = torch.tensor(self.rewards[index])
        action = torch.tensor(self.actions[index])
        ob = torch.tensor(self.obs[index])
        done = torch.tensor(self.dones[index])
        return ob, action, reward, done


class Critic(eqx.Module):
    """Critic network for the policy gradient algorithm."""

    layers: list

    def __init__(self, layers: list[int], key: PRNGKeyArray):
        super().__init__()
        self.layers = []
        for i in range(len(layers) - 1):
            key, subkey = jax.random.split(key)
            self.layers.append(eqx.nn.Linear(layers[i], layers[i + 1], key=subkey))
            del subkey
            if i != len(layers) - 2:
                self.layers.append(jax.nn.relu)

    def __call__(self, x: Array) -> Array:
        for layer in self.layers:
            x = layer(x)
        return x


class Policy(eqx.Module):
    """Policy network for the policy gradient algorithm."""

    layers: list

    def __init__(self, layers: list[int], key: PRNGKeyArray):
        super().__init__()
        self.layers = []
        for i in range(len(layers) - 1):
            key, subkey = jax.random.split(key)
            self.layers.append(eqx.nn.Linear(layers[i], layers[i + 1], key=subkey))
            del subkey
            if i != len(layers) - 2:
                self.layers.append(jax.nn.relu)

    def __call__(self, x: Array) -> Array:
        for layer in self.layers:
            x = layer(x)
        return x


def calculate_gae(
    rewards: Array,
    values: Array,
    dones: Array,
    gamma: Float32,
    lam: Float32,
) -> Array:
    values = jnp.append(values, values[0])
    gae = 0
    advantages = jnp.zeros_like(rewards)
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages = advantages.at[t].set(gae)

    return advantages


@eqx.filter_jit
def get_action(state: Array, policy: PyTree, epsilon: float, key: PRNGKeyArray) -> int:
    """Get action from the policy network."""
    key, subkey1, subkey2 = jax.random.split(key, 3)
    logits = policy(state)

    def follow_policy():
        action = tfp.distributions.Categorical(logits=logits).sample(seed=subkey1)
        return action

    def explore():
        action = jax.random.randint(subkey2, shape=(), minval=0, maxval=len(logits))
        return action

    action = jax.lax.cond(jax.random.uniform(key) > epsilon, follow_policy, explore)

    return action


def rollout_discrete(
    env: gym.Env, policy: PyTree, epsilon: float, key: PRNGKeyArray
) -> Tuple[Array, Array, Array, Array]:
    """Perform a rollout using the policy network."""
    states = []
    actions = []
    rewards = []
    dones = []

    state, info = env.reset()

    while True:
        states.append(state)
        key, subkey = jax.random.split(key)

        action = get_action(state, policy, epsilon, subkey)
        del subkey

        state, reward, terminated, truncated, info = env.step(int(action))

        actions.append(action)
        rewards.append(reward)
        if terminated or truncated:
            dones.append(1)
            break
        else:
            dones.append(0)
    states = jnp.stack(states)
    actions = jnp.array(actions)
    rewards = jnp.array(rewards)
    dones = jnp.array(dones)
    return states, actions, rewards, dones


def get_baseline_of_trajectory(advantages: Array) -> Array:
    """Get the baseline of a trajectory."""
    return jnp.mean(advantages)


@eqx.filter_jit
def critic_loss(
    critic: PyTree,
    states: Array,
    rewards: Array,
) -> Array:
    """Get the critic loss."""
    values = jax.vmap(critic)(states)
    loss = jnp.mean((values - rewards) ** 2)
    return loss


@eqx.filter_jit
def critic_step(
    critic: PyTree,
    states: Array,
    rewards: Array,
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
) -> Tuple[PyTree, optax.OptState, Array]:
    """Perform a step of the critic network."""
    loss, grads = eqx.filter_value_and_grad(critic_loss)(critic, states, rewards)
    updates, opt_state = optimizer.update(grads, opt_state, critic)
    critic = eqx.apply_updates(critic, updates)
    return critic, opt_state, loss  # type: ignore


@eqx.filter_jit
def loss_fn(
    policy: PyTree,
    states: Array,
    actions: Array,
    rewards: Array,
    dones: Array,
    gamma: float,
    critic: PyTree,
    lam=0.95,
) -> Array:
    logits = jax.vmap(policy)(states)
    values = jax.vmap(critic)(states)
    advantages = calculate_gae(rewards, values, dones, gamma, lam)
    loss = rlh.get_policy_gradient_discrete_loss(logits, actions, advantages)
    return loss


@eqx.filter_jit
def step(
    states: Array,
    actions: Array,
    rewards: Array,
    dones: Array,
    policy: PyTree,
    gamma: float,
    optimizer: optax.GradientTransformation,
    opt_state: optax.Params,
    critic: PyTree,
):
    """Perform a single step of the policy gradient algorithm."""
    loss, grads = eqx.filter_value_and_grad(loss_fn)(
        policy, states, actions, rewards, dones, gamma, critic
    )
    updates, opt_state = optimizer.update(grads, opt_state, policy)
    policy = eqx.apply_updates(policy, updates)
    return policy, opt_state, loss


def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size), "valid") / window_size


def experiment(hyperparams: dict):
    """Run the experiment."""

    env = hyperparams["env"]
    learning_rate = hyperparams["learning_rate"]
    gamma = hyperparams["gamma"]
    layers = hyperparams["layers"]
    epsilon = hyperparams["epsilon"]
    critic_learning_rate = hyperparams["critic_learning_rate"]
    critic_layers = hyperparams["critic_layers"]

    policy = Policy(layers, jax.random.PRNGKey(38))
    optimizer = optax.adamw(learning_rate)
    opt_state = eqx_helpers.eqx_init_optimiser(optimizer, policy)

    critic = Critic(critic_layers, jax.random.PRNGKey(39))
    critic_optimizer = optax.adamw(critic_learning_rate)
    critic_opt_state = eqx_helpers.eqx_init_optimiser(critic_optimizer, critic)

    all_rewards = []
    all_losses = []
    key = jax.random.PRNGKey(42)
    for i in tqdm(range(5000)):
        key, subkey = jax.random.split(key)

        states, actions, rewards, dones = rollout_discrete(env, policy, epsilon, subkey)
        del subkey

        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        dones = np.array(dones)

        all_rewards.append(np.sum(rewards))

        dataset = RLDataset(states, actions, rewards, dones)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=False, drop_last=True)

        losses = []
        for batch in dataloader:
            states, actions, rewards, dones = batch

            states = jnp.array(states.numpy())
            actions = jnp.array(actions.numpy())
            rewards = jnp.array(rewards.numpy())
            dones = jnp.array(dones.numpy())

            critic, critic_opt_state, loss = critic_step(
                critic, states, rewards, critic_opt_state, critic_optimizer
            )
            losses.append(loss)

            policy, opt_state, loss = step(
                states,
                actions,
                rewards,
                dones,
                policy,
                gamma,
                optimizer,
                opt_state,
                critic,
            )
        all_losses.append(np.mean(losses))
        if i % 500 == 0 and i >= 100:
            print(
                f"Episode {i} | Moving average reward (last 100): {moving_average(all_rewards, 100)[-1]}, Moving average loss (last 100): {moving_average(all_losses, 100)[-1]}"  # noqa: E501
            )

    mean_reward = np.mean(all_rewards)
    return mean_reward


def objective(trial: optuna.Trial, env_name: str):
    env = gym.make(env_name)
    obs_space = env.observation_space.shape[0]
    action_space = env.action_space.n

    learning_rate = trial.suggest_float("learning_rate", 0.000001, 0.001)
    gamma = trial.suggest_float("gamma", 0.8, 0.999)
    epsilon = trial.suggest_float("epsilon", 0.001, 0.3)
    hidden_layers = trial.suggest_int("hidden_layers", 0, 5)

    critic_learning_rate = trial.suggest_float("critic_learning_rate", 0.00001, 0.001)
    critic_hidden_layers = trial.suggest_int("critic_hidden_layers", 0, 5)

    layers = [obs_space]
    critic_layers = [obs_space]
    for i in range(hidden_layers):
        layers.append(trial.suggest_int(f"hidden_layer_{i}", 1, 128))
    for i in range(critic_hidden_layers):
        critic_layers.append(trial.suggest_int(f"critic_hidden_layer_{i}", 1, 128))
    layers.append(action_space)
    critic_layers.append(1)

    hyperparams = {
        "env": env,
        "learning_rate": learning_rate,
        "gamma": gamma,
        "layers": layers,
        "epsilon": epsilon,
        "critic_learning_rate": critic_learning_rate,
        "critic_layers": critic_layers,
    }
    return experiment(hyperparams)


def run_trials():
    # env_name = "LunarLander-v2"
    env_name = "CartPole-v1"
    current_file_path = pathlib.Path(__file__).parent.absolute()
    database_url = current_file_path.parent / "studies/studies.db"
    study = optuna.create_study(
        study_name=env_name,
        direction="maximize",
        storage=f"sqlite:///{database_url}",
        load_if_exists=True,
    )
    study.optimize(lambda trial: objective(trial, env_name), n_trials=50, n_jobs=1)


def main():
    # env_name = "LunarLander-v2"
    env_name = "CartPole-v1"
    env = gym.make(env_name)
    obs_space = env.observation_space.shape[0]
    action_space = env.action_space.n

    layers = [obs_space, action_space]

    learning_rate = 0.001
    gamma = 0.99
    epsilon = 0.0
    hyperparams = {
        "env": env,
        "learning_rate": learning_rate,
        "gamma": gamma,
        "layers": layers,
        "epsilon": epsilon,
    }
    experiment(hyperparams)


if __name__ == "__main__":
    run_trials()
    # main()
