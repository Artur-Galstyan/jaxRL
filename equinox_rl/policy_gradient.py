import equinox as eqx
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow_probability.substrates.jax as tfp
from jaxtyping import Array, Float32, PRNGKeyArray, PyTree
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from equinox_rl.common import eqx_helpers, gym_helpers, rl_helpers
import matplotlib.pyplot as plt


class Policy(eqx.Module):
    """Policy network for the policy gradient algorithm in a discrete action space."""

    layers: list

    def __init__(self, layers: list[int], key: PRNGKeyArray) -> None:
        """Initialize the policy network.
        Args:
            layers: The layers of the policy network.
            key: The key to use for initialization.
        """
        self.layers = []

        for i in range(len(layers) - 1):
            key, subkey = jax.random.split(key)
            self.layers.append(eqx.nn.Linear(layers[i], layers[i + 1], key=subkey))
            del subkey

            # Add a ReLU activation function to all but the last layer.
            if i < len(layers) - 2:
                self.layers.append(jax.nn.relu)

    def __call__(self, x: Float32[Array, "state_dims"]) -> Array:
        """Forward pass of the policy network.
        Args:
            x: The input to the policy network.
        Returns:
            The output of the policy network.
        """
        for layer in self.layers:
            x = layer(x)
        return x


@eqx.filter_jit
def get_action(state: Float32[Array, "state_dims"], policy: Policy, key: PRNGKeyArray):
    """Get an action from the policy network.
    Args:
        state: The state to get an action from.
        policy: The policy network.
        key: The key to use for sampling.
    Returns:
        The action sampled from the policy network.
    """
    key, subkey = jax.random.split(key)
    logits = policy(state)
    action = tfp.distributions.Categorical(logits=logits).sample(seed=subkey)

    return action


def loss_fn(
    policy: PyTree,
    states: Float32[Array, "n_steps state_dims"],
    actions: Float32[Array, "n_steps"],
    rewards: Float32[Array, "n_steps"],
    dones: Float32[Array, "n_steps"],
    gamma: float,
) -> Array:
    logits = eqx.filter_vmap(policy)(states)
    advantages = rl_helpers.get_total_discounted_rewards(rewards, gamma)

    loss = rl_helpers.get_policy_gradient_discrete_loss(logits, actions, advantages)
    return loss


@eqx.filter_jit
def step(
    states: Float32[Array, "n_steps state_dims"],
    actions: Float32[Array, "n_steps"],
    rewards: Float32[Array, "n_steps"],
    dones: Float32[Array, "n_steps"],
    gamma: float,
    optimizer: optax.GradientTransformation,
    opt_state: PyTree,
    policy: PyTree,
):
    loss, grads = eqx.filter_value_and_grad(loss_fn)(
        policy, states, actions, rewards, dones, gamma
    )
    updates, opt_state = optimizer.update(grads, opt_state, policy)
    policy = eqx.apply_updates(policy, updates)

    return policy, opt_state, loss


def train(
    states: Float32[Array, "n_steps state_dims"],
    actions: Float32[Array, "n_steps"],
    rewards: Float32[Array, "n_steps"],
    dones: Float32[Array, "n_steps"],
    gamma: float,
    policy: PyTree,
    opt_state: PyTree,
    optimizer: optax.GradientTransformation,
) -> tuple[PyTree, PyTree, np.float32]:
    """Train the policy network."""

    # Convert jax arrays to numpy arrays for the dataloader
    states = np.array(states)  # type: ignore
    actions = np.array(actions)  # type: ignore
    rewards = np.array(rewards)  # type: ignore
    dones = np.array(dones)  # type: ignore
    dataset = gym_helpers.RLDataset(states, actions, rewards, dones)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, drop_last=True)

    losses = []

    for batch in dataloader:
        b_states, b_actions, b_rewards, b_dones = batch

        b_states = jnp.array(b_states.numpy())
        b_actions = jnp.array(b_actions.numpy())
        b_rewards = jnp.array(b_rewards.numpy())
        b_dones = jnp.array(b_dones.numpy())

        policy, opt_state, loss = step(
            b_states, b_actions, b_rewards, b_dones, gamma, optimizer, opt_state, policy
        )

        losses.append(loss)

    return policy, opt_state, np.mean(losses)


def experiment(hyperparams: dict):
    """Run the experiment."""
    env_name = hyperparams["env_name"]
    learning_rate = hyperparams["learning_rate"]
    gamma = hyperparams["gamma"]
    layers = hyperparams["layers"]
    n_episodes = hyperparams["n_episodes"]

    policy = Policy(layers, key=jax.random.PRNGKey(0))
    optimizer = optax.adamw(learning_rate)
    opt_state = eqx_helpers.eqx_init_optimiser(optimizer, policy)

    key = jax.random.PRNGKey(42)
    env = gym.make(env_name)

    losses = []
    all_rewards = []
    for i in tqdm(range(n_episodes)):
        key, subkey = jax.random.split(key)

        states, actions, rewards, dones = gym_helpers.rollout_discrete(
            env, get_action, {"policy": policy}, key=subkey
        )

        del subkey

        all_rewards.append(np.sum(rewards))

        policy, opt_state, loss = train(
            states, actions, rewards, dones, gamma, policy, opt_state, optimizer
        )

        losses.append(loss)

        if (i % 100 == 0 and i >= 100) or i == n_episodes - 1:
            print(
                f"Episode {i} | R: {gym_helpers.moving_average(all_rewards, 100)[-1]}"
            )
            print(f"Episode {i} | L: {gym_helpers.moving_average(losses, 100)[-1]}")

    mean_reward = np.mean(all_rewards)

    return mean_reward


def main() -> None:
    """Run the experiment."""
    env = gym.make("LunarLander-v2")
    if env.observation_space.shape is None:
        raise ValueError("Observation space shape is None")
    if env.action_space.n is None:  # type: ignore
        raise ValueError("Action space n is None")

    state_dims = env.observation_space.shape[0]
    n_actions = env.action_space.n  # type: ignore

    hyperparams = {
        "env_name": "LunarLander-v2",
        "learning_rate": 1e-4,
        "gamma": 0.99,
        "layers": [state_dims, 256, 128, 64, n_actions],
        "n_episodes": 10000,
    }

    experiment(hyperparams)


if __name__ == "__main__":
    main()
