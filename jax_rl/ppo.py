from __future__ import print_function
from typing import Optional, Tuple

import equinox as eqx
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import rlax
from beartype import beartype
from jaxtyping import Array, Float32, PRNGKeyArray, PyTree
from torch.utils.data import DataLoader
from tqdm import tqdm

from jax_rl.common import gym_helpers, rl_helpers


class Actor(eqx.Module):
    """Actor network for the policy gradient algorithm in a discrete action space."""

    mlp: eqx.nn.MLP

    def __init__(
        self,
        in_size: int,
        out_size: int,
        key: PRNGKeyArray,
        width_size: int = 32,
        depth: int = 2,
    ) -> None:
        key, subkey = jax.random.split(key)
        self.mlp = eqx.nn.MLP(
            in_size=in_size,
            out_size=out_size,
            width_size=width_size,
            depth=depth,
            key=subkey,
        )

    def __call__(self, x: Float32[Array, "state_dims"] | np.ndarray) -> Array:
        """Forward pass of the actor network.
        Args:
            x: The input to the actor network.
        Returns:
            The output of the actor network.
        """
        return self.mlp(x)


class Critic(eqx.Module):
    """Critic network for the policy gradient algorithm in a discrete action space."""

    mlp: eqx.nn.MLP

    def __init__(
        self,
        in_size: int,
        out_size: int,
        key: PRNGKeyArray,
        width_size: int = 64,
        depth: int = 3,
    ) -> None:
        key, subkey = jax.random.split(key)
        self.mlp = eqx.nn.MLP(
            in_size=in_size,
            out_size=out_size,
            width_size=width_size,
            depth=depth,
            key=subkey,
        )

    def __call__(self, x: Float32[Array, "state_dims"] | np.ndarray) -> Array:
        """Forward pass of the critic network.
        Args:
            x: The input to the critic network.
        Returns:
            The output of the critic network.
        """
        return self.mlp(x)


@eqx.filter_jit
def get_action(obs: Array, actor: Actor, key: PRNGKeyArray) -> tuple[int, Array]:
    """Sample an action from the policy network.
    Args:
        obs: The observation from the environment.
        key: The random key.
        actor: The policy network.
    Returns:
        The action and the logits.
    """
    logits = actor(obs)
    key, subkey = jax.random.split(key)
    action = jax.random.categorical(subkey, logits)
    return action, logits


def objective_fn(
    actor: PyTree,
    states: Array,
    actions: Array,
    rewards: Array,
    log_probs: Array,
    critic: PyTree,
) -> Array:
    values = jax.vmap(critic)(states).reshape(-1)
    logits = jax.vmap(actor)(states)
    new_log_probs = jax.nn.log_softmax(logits)
    new_log_probs = jnp.take_along_axis(
        new_log_probs, jnp.expand_dims(actions, -1), axis=1
    ).reshape(-1)
    rewards_to_go = rl_helpers.get_total_discounted_rewards(rewards)
    advantages = rewards_to_go - values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    clipped_surrogate_pg_loss = rlax.clipped_surrogate_pg_loss(
        prob_ratios_t=jax.numpy.exp(new_log_probs - log_probs),
        adv_t=advantages,
        epsilon=0.2,
        use_stop_gradient=True,
    )

    return clipped_surrogate_pg_loss


@eqx.filter_jit
def step(
    actor: PyTree,
    states: Float32[Array, "n_steps state_dim"],
    actions: Float32[Array, "n_steps"],
    log_probs: Float32[Array, "n_steps"],
    rewards: Float32[Array, "n_steps"],
    optimiser: optax.GradientTransformation,
    optimiser_state: optax.OptState,
    critic: PyTree,
) -> Tuple[PyTree, optax.OptState]:
    _, grad = eqx.filter_value_and_grad(objective_fn)(
        actor, states, actions, rewards, log_probs, critic
    )
    updates, optimiser_state = optimiser.update(grad, optimiser_state, actor)
    actor = eqx.apply_updates(actor, updates)

    return actor, optimiser_state


@eqx.filter_jit
def step_critic(
    critic: PyTree,
    states: Float32[Array, "n_steps state_dim"],
    rewards: Float32[Array, "n_steps"],
    optimiser: optax.GradientTransformation,
    optimiser_state: optax.OptState,
) -> Tuple[PyTree, optax.OptState]:
    def mse(critic, states, rewards):
        values = jax.vmap(critic)(states)
        return jnp.mean((values - rewards) ** 2)

    _, grad = eqx.filter_value_and_grad(mse)(critic, states, rewards)
    updates, optimiser_state = optimiser.update(grad, optimiser_state, critic)
    critic = eqx.apply_updates(critic, updates)

    return critic, optimiser_state


def train(
    env: gym.Env,
    env_type: str,
    actor: Actor,
    critic: Critic,
    actor_optimiser: optax.GradientTransformation,
    critic_optimiser: optax.GradientTransformation,
    obs_size: int,
    action_size: int,
    env_params: dict = {},
    n_epochs: int = 30,
    n_episodes: int = 1000,
    updates_per_batch: int = 1,
    key: PRNGKeyArray = jax.random.PRNGKey(0),
    max_steps: Optional[int] = None,
    render: bool = False,
) -> Actor:
    """Train the policy network.
    Args:
        env: The gym environment.
        actor: The policy network.
        critic: The value network.
        actor_optimiser: The optimiser for the policy network.
        critic_optimiser: The optimiser for the value network.
        n_epochs: The number of epochs to train for.
        n_episodes: The number of episodes to run for each epoch.
        key: The random key.
        max_steps: The maximum number of steps to run for each episode.
        render: Whether to render the environment.
    Returns:
        The trained policy network.
    """
    key, policy_key, value_key = jax.random.split(key, 3)
    if actor is None:
        actor = Actor(
            in_size=int(obs_size),
            out_size=int(action_size),
            key=policy_key,
        )
    if critic is None:
        critic = Critic(
            in_size=int(obs_size),
            out_size=1,
            key=value_key,
        )
    assert actor is not None, "Actor was not initialised"
    assert critic is not None, "Critic was not initialised"
    opt_state_critic = critic_optimiser.init(eqx.filter(critic, eqx.is_inexact_array))
    opt_state_actor = actor_optimiser.init(eqx.filter(actor, eqx.is_array))
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
            if env_type == "gymnax":
                dataset, _ = gym_helpers.gymnax_rollout_discrete(
                    env, env_params, get_action, {"actor": actor}, subkey
                )
            else:
                dataset, _ = gym_helpers.rollout_discrete(
                    env, get_action, {"actor": actor}, subkey
                )
            dataloader = DataLoader(
                dataset, batch_size=4, shuffle=False, drop_last=True
            )

            epoch_rewards += jnp.sum(dataset.rewards.numpy())

            for batch in dataloader:
                for i in range(updates_per_batch):
                    b_states, b_actions, b_rewards, b_log_probs, b_dones = batch
                    b_states = jnp.array(b_states.numpy())
                    b_actions = jnp.array(b_actions.numpy())
                    b_log_probs = jnp.array(b_log_probs.numpy())
                    b_rewards = jnp.array(b_rewards.numpy())
                    b_dones = jnp.array(b_dones.numpy())

                    critic, opt_state_critic = step_critic(
                        critic=critic,
                        states=b_states,
                        rewards=b_rewards,
                        optimiser=critic_optimiser,
                        optimiser_state=opt_state_critic,
                    )

                    actor, opt_state_actor = step(
                        actor=actor,
                        states=b_states,
                        actions=b_actions,
                        rewards=b_rewards,
                        log_probs=b_log_probs,
                        optimiser=actor_optimiser,
                        optimiser_state=opt_state_actor,
                        critic=critic,
                    )
        rewards_to_show.append(jnp.mean(epoch_rewards / n_episodes))
        reward_log.set_description_str(f"Last avg. rewards: {rewards_to_show[-1]}")

    return actor, critic, rewards_to_show
