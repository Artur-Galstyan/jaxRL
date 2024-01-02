from typing import Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import rlax
from jaxtyping import Array, PRNGKeyArray, PyTree

from jax_rl.commons import ReplayBuffer


def objective_fn(
    policy: PyTree,
    critic: PyTree,
    replay_buffer: ReplayBuffer,
    lambda_: float = 1.0,
    gamma: float = 0.99,
) -> Array:
    logits = eqx.filter_vmap(policy)(replay_buffer.states)
    log_probs = jax.nn.log_softmax(logits)
    log_probs_actions = jnp.take_along_axis(
        log_probs, jnp.expand_dims(replay_buffer.actions, -1), axis=1
    )
    rewards = rlax.lambda_returns(
        r_t=replay_buffer.rewards,
        discount_t=jnp.array([gamma] * len(replay_buffer.rewards)),
        v_t=eqx.filter_vmap(critic)(replay_buffer.states).reshape(-1),
        lambda_=lambda_,
    )
    values = eqx.filter_vmap(critic)(replay_buffer.states)
    advantages = rewards - values
    return -jnp.mean(log_probs_actions * advantages)


def step_actor_network(
    actor: PyTree,
    optimiser: optax.GradientTransformation,
    optimiser_state: optax.OptState,
    critic: PyTree,
    replay_buffer: ReplayBuffer,
) -> Tuple[PyTree, optax.OptState]:
    grad = eqx.filter_grad(objective_fn)(actor, critic, replay_buffer)
    updates, optimiser_state = optimiser.update(grad, optimiser_state, actor)
    actor = eqx.apply_updates(actor, updates)

    return actor, optimiser_state


def critic_loss_fn(
    critic: PyTree,
    replay_buffer: ReplayBuffer,
) -> Array:
    """Calculate the value loss for a given set of states and rewards.
    Args:
        value_network: The value network.
        states: The states to calculate the value loss for.
        rewards: The rewards to calculate the value loss for.
    Returns:
        The value loss.
    """
    values = eqx.filter_vmap(critic)(replay_buffer.states)
    return jnp.mean((values - replay_buffer.rewards) ** 2)


def step_critic_network(
    critic: PyTree,
    optimiser: optax.GradientTransformation,
    optimiser_state: optax.OptState,
    replay_buffer: ReplayBuffer,
) -> Tuple[PyTree, optax.OptState]:
    grad = eqx.filter_grad(critic_loss_fn)(critic, replay_buffer)
    updates, optimiser_state = optimiser.update(grad, optimiser_state, critic)
    critic = eqx.apply_updates(critic, updates)

    return critic, optimiser_state


@eqx.filter_jit
def train(
    actor: PyTree,
    actor_optimiser: optax.GradientTransformation,
    actor_optimiser_state: optax.OptState,
    critic: PyTree,
    critic_optimiser: optax.GradientTransformation,
    critic_optimiser_state: optax.OptState,
    replay_buffer: ReplayBuffer,
    *,
    key: PRNGKeyArray,
) -> (PyTree, PyTree, Array):
    key, subkey = jax.random.split(key)

    actor, actor_optimiser_state = step_actor_network(
        actor=actor,
        optimiser=actor_optimiser,
        optimiser_state=actor_optimiser_state,
        critic=critic,
        replay_buffer=replay_buffer,
    )
    critic, critic_optimiser_state = step_critic_network(
        critic=critic,
        optimiser=critic_optimiser,
        optimiser_state=critic_optimiser_state,
        replay_buffer=replay_buffer,
    )

    return actor, actor_optimiser_state, critic, critic_optimiser_state
