from typing import Optional, Tuple

import equinox as eqx
import gymnasium
import gymnax
import jax
import jax.numpy as jnp
import optax
from beartype import beartype
from gymnax.environments.environment import EnvParams
from jaxtyping import Array, Float32, PRNGKeyArray, PyTree
from torch.utils.data import DataLoader
from tqdm import tqdm

from jax_rl import common_helper_functions
from jax_rl.common import gym_helpers, rl_helpers


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
    critic: PyTree,
):
    logits = eqx.filter_vmap(policy)(states)
    log_probs = jax.nn.log_softmax(logits)
    log_probs_actions = jnp.take_along_axis(
        log_probs, jnp.expand_dims(actions, -1), axis=1
    )
    rewards = rl_helpers.get_total_discounted_rewards(rewards)
    values = eqx.filter_vmap(critic)(states)
    advantages = rewards - values
    return -jnp.mean(log_probs_actions * advantages)


@eqx.filter_jit
def step_actor_network(
    actor: PyTree,
    states: Float32[Array, "n_steps state_dim"],
    actions: Float32[Array, "n_steps"],
    rewards: Float32[Array, "n_steps"],
    optimiser: optax.GradientTransformation,
    optimiser_state: optax.OptState,
    critic: PyTree,
) -> Tuple[PyTree, optax.OptState]:
    _, grad = eqx.filter_value_and_grad(objective_fn)(
        actor, states, actions, rewards, critic
    )
    updates, optimiser_state = optimiser.update(grad, optimiser_state, actor)
    actor = eqx.apply_updates(actor, updates)

    return actor, optimiser_state


def critic_loss_fn(
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
def step_critic_network(
    critic: PyTree,
    states: Float32[Array, "n_steps state_dim"],
    rewards: Float32[Array, "n_steps"],
    optimiser: optax.GradientTransformation,
    optimiser_state: optax.OptState,
) -> Tuple[PyTree, optax.OptState]:
    _, grad = eqx.filter_value_and_grad(critic_loss_fn)(critic, states, rewards)
    updates, optimiser_state = optimiser.update(grad, optimiser_state, critic)
    critic = eqx.apply_updates(critic, updates)

    return critic, optimiser_state


def train(
    env,
    obs_size: int,
    action_size: int,
    actor_optimiser: optax.GradientTransformation,
    critic_optimiser: optax.GradientTransformation,
    env_params: Optional[EnvParams] = None,
    actor: Optional[PyTree] = None,
    critic: Optional[PyTree] = None,
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
    key, policy_key, value_key = jax.random.split(key, 3)
    if actor is None:
        actor = Policy(
            in_size=obs_size,
            out_size=action_size,
            key=policy_key,
        )
    if critic is None:
        critic = ValueNetwork(
            in_size=obs_size,
            out_size=1,
            key=value_key,
        )
    assert actor is not None, "Policy was not initialised"
    assert critic is not None, "Value network was not initialised"
    opt_state_critic = critic_optimiser.init(eqx.filter(critic, eqx.is_inexact_array))
    opt_state_actor = actor_optimiser.init(eqx.filter(actor, eqx.is_inexact_array))
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
            dataset, _ = common_helper_functions.rollout_discrete(
                env, get_action, {"policy": actor}, subkey, env_params=env_params
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

                value_network, opt_state_value = step_critic_network(
                    critic=critic,
                    states=b_states,
                    rewards=b_rewards,
                    optimiser=critic_optimiser,
                    optimiser_state=opt_state_critic,
                )

                policy, opt_state = step_actor_network(
                    actor=actor,
                    states=b_states,
                    actions=b_actions,
                    rewards=b_rewards,
                    optimiser=actor_optimiser,
                    optimiser_state=opt_state_actor,
                    critic=critic,
                )
        rewards_to_show.append(jnp.mean(epoch_rewards / n_episodes))
        reward_log.set_description_str(f"Last avg. rewards: {rewards_to_show[-1]}")

    return actor, critic, rewards_to_show
