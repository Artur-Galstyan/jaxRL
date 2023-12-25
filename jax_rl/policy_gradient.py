from typing import Optional, Tuple

import equinox as eqx
import gymnasium
import jax
import jax.numpy as jnp

import optax
from jaxtyping import Array, Float32, PRNGKeyArray, PyTree
from torch.utils.data import DataLoader
from tqdm import tqdm

from jax_rl.common import gym_helpers, rl_helpers
from beartype import beartype


@beartype
class Policy(eqx.Module):
    """Policy network for the policy gradient algorithm in a discrete action space."""

    mlp: eqx.nn.MLP

    def __init__(
        self,
        in_size: int,
        out_size: int,
        key: PRNGKeyArray,
        width_size: int = 32,
        depth: int = 2,
    ) -> None:
        key, *subkeys = jax.random.split(key, 5)
        self.mlp = eqx.nn.MLP(
            in_size=in_size,
            out_size=out_size,
            width_size=width_size,
            depth=depth,
            key=key,
        )

    def __call__(self, x: Float32[Array, "state_dims"]) -> Array:
        """Forward pass of the policy network.
        Args:
            x: The input to the policy network.
        Returns:
            The output of the policy network.
        """
        return self.mlp(x)


@beartype
class ValueNetwork(eqx.Module):
    """Value network for the policy gradient algorithm in a discrete action space."""

    mlp: eqx.nn.MLP

    def __init__(
        self,
        in_size: int,
        out_size: int,
        key: PRNGKeyArray,
        width_size: int = 64,
        depth: int = 3,
    ) -> None:
        key, *subkeys = jax.random.split(key, 5)
        self.mlp = eqx.nn.MLP(
            in_size=in_size,
            out_size=out_size,
            width_size=width_size,
            depth=depth,
            key=key,
        )

    def __call__(self, x: Float32[Array, "state_dims"]) -> Array:
        """Forward pass of the policy network.
        Args:
            x: The input to the policy network.
        Returns:
            The output of the policy network.
        """
        return self.mlp(x)


@eqx.filter_jit
def get_action(
    state: Float32[Array, "state_dims"],
    policy: Policy,
    key: PRNGKeyArray,
) -> Array:
    """Get an action from the policy network.
    Args:
        state: The state to get an action from.
        policy: The policy network.
        key: The key to use for sampling.
    Returns:
        The action sampled from the policy network.
        The logits from the policy network.
    """
    logits = policy(state)
    action = jax.random.categorical(key, logits)
    return action, logits


def objective_fn(
    policy: PyTree,
    states: Float32[Array, "n_steps state_dim"],
    actions: Float32[Array, "n_steps"],
    rewards: Float32[Array, "n_steps"],
    value_network: PyTree,
):
    logits = eqx.filter_vmap(policy)(states)
    log_probs = jax.nn.log_softmax(logits)
    log_probs_actions = jnp.take_along_axis(
        log_probs, jnp.expand_dims(actions, -1), axis=1
    )
    rewards = rl_helpers.get_total_discounted_rewards(rewards)
    values = eqx.filter_vmap(value_network)(states)
    advantages = rewards - values
    return -jnp.mean(log_probs_actions * advantages)


@eqx.filter_jit
def step(
    policy: PyTree,
    states: Float32[Array, "n_steps state_dim"],
    actions: Float32[Array, "n_steps"],
    rewards: Float32[Array, "n_steps"],
    optimiser: optax.GradientTransformation,
    optimiser_state: optax.OptState,
    value_network: PyTree,
) -> Tuple[PyTree, optax.OptState]:
    _, grad = eqx.filter_value_and_grad(objective_fn)(
        policy, states, actions, rewards, value_network
    )
    updates, optimiser_state = optimiser.update(grad, optimiser_state, policy)
    policy = eqx.apply_updates(policy, updates)

    return policy, optimiser_state


def value_loss_fn(
    value_network: PyTree,
    states: Float32[Array, "n_steps state_dim"],
    rewards: Float32[Array, "n_steps"],
) -> Array:
    """Calculate the value loss for a given set of states and rewards.
    Args:
        value_network: The value network.
        states: The states to calculate the value loss for.
        rewards: The rewards to calculate the value loss for.
    Returns:
        The value loss.
    """
    values = eqx.filter_vmap(value_network)(states)
    return jnp.mean((values - rewards) ** 2)


@eqx.filter_jit
def step_value_network(
    value_network: PyTree,
    states: Float32[Array, "n_steps state_dim"],
    rewards: Float32[Array, "n_steps"],
    optimiser: optax.GradientTransformation,
    optimiser_state: optax.OptState,
) -> Tuple[PyTree, optax.OptState]:
    _, grad = eqx.filter_value_and_grad(value_loss_fn)(value_network, states, rewards)
    updates, optimiser_state = optimiser.update(grad, optimiser_state, value_network)
    value_network = eqx.apply_updates(value_network, updates)

    return value_network, optimiser_state


def train(
    env: gymnasium.Env,
    optimiser: optax.GradientTransformation,
    policy: Optional[PyTree] = None,
    value_network: Optional[PyTree] = None,
    n_epochs: int = 30,
    n_episodes: int = 1000,
    *,
    key: PRNGKeyArray,
) -> (PyTree, PyTree, Array):
    """Train a policy network using the policy gradient algorithm.
    Args:
        env: The environment to train on.
        optimiser: The optimiser to use.
        policy: The policy network to train. If None, a new one will be initialised.
        value_network: The value network to train. If None, a new one will be initialised.
        n_epochs: The number of epochs to train for.
        n_episodes: The number of episodes to train for.
    Returns:
        The trained policy network.
        The trained value network.
        The rewards per epoch.
    """
    assert isinstance(
        env.action_space, gymnasium.spaces.Discrete
    ), "Only discrete action spaces are supported."
    assert isinstance(
        env.observation_space, gymnasium.spaces.Box
    ), "Only box observation spaces are supported."
    key, policy_key, value_key = jax.random.split(key, 3)
    if policy is None:
        policy = Policy(
            in_size=int(env.observation_space.shape[0]),
            out_size=int(env.action_space.n),
            key=policy_key,
        )
    if value_network is None:
        value_network = ValueNetwork(
            in_size=int(env.observation_space.shape[0]),
            out_size=1,
            key=value_key,
        )
    assert policy is not None, "Policy was not initialised"
    assert value_network is not None, "Value network was not initialised"
    opt_state_value = optimiser.init(eqx.filter(value_network, eqx.is_inexact_array))
    opt_state = optimiser.init(eqx.filter(policy, eqx.is_array))
    key, subkey = jax.random.split(key)
    reward_log = tqdm(
        total=n_epochs,
        desc="Reward",
        position=2,
        bar_format="{desc}",
    )
    rewards_to_show = []
    for _ in tqdm(range(n_epochs), desc="Epochs", position=0):
        epoch_rewards = 0
        for _ in tqdm(range(n_episodes), desc="Episodes", position=1, leave=False):
            key, subkey = jax.random.split(key)
            dataset, _ = gym_helpers.rollout_discrete(
                env, get_action, {"policy": policy}, subkey
            )
            dataloader = DataLoader(
                dataset, batch_size=4, shuffle=False, drop_last=True
            )

            epoch_rewards += jnp.sum(dataset.rewards.numpy())

            for batch in dataloader:
                b_states, b_actions, b_rewards, b_log_probs, b_dones = batch
                b_states = jnp.array(b_states.numpy())
                b_actions = jnp.array(b_actions.numpy())
                b_log_probs = jnp.array(b_log_probs.numpy())
                b_rewards = jnp.array(b_rewards.numpy())
                b_dones = jnp.array(b_dones.numpy())

                value_network, opt_state_value = step_value_network(
                    value_network=value_network,
                    states=b_states,
                    rewards=b_rewards,
                    optimiser=optimiser,
                    optimiser_state=opt_state_value,
                )

                policy, opt_state = step(
                    policy=policy,
                    states=b_states,
                    actions=b_actions,
                    rewards=b_rewards,
                    optimiser=optimiser,
                    optimiser_state=opt_state,
                    value_network=value_network,
                )
        rewards_to_show.append(jnp.mean(epoch_rewards / n_episodes))
        reward_log.set_description_str(f"Last avg. rewards: {rewards_to_show[-1]}")

    return policy, value_network, rewards_to_show
