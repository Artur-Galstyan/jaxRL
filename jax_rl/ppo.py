from typing import Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import rlax
from jaxtyping import Array, PRNGKeyArray, PyTree

from jax_rl.commons import ReplayBuffer, calculate_gae


def objective_fn(
    policy: PyTree,
    critic: PyTree,
    replay_buffer: ReplayBuffer,
    epsilon: float = 0.2,
    gamma: float = 0.99,
    lambda_: float = 0.95,
) -> Array:
    # advantages only go from 0 to T-1, because the advantage at T is 0 by definition
    # so we need to pad the advantages with a zero at the end
    # advantages = rlax.truncated_generalized_advantage_estimation(
    #     r_t=replay_buffer.rewards[1:],
    #     discount_t=jnp.array([gamma] * (len(replay_buffer.rewards) - 1)),
    #     values=eqx.filter_vmap(critic)(replay_buffer.states).reshape(-1),
    #     lambda_=lambda_,
    # )
    # advantages = jnp.concatenate([advantages, jnp.array([0.0])])
    advantages = calculate_gae(
        rewards=replay_buffer.rewards,
        values=eqx.filter_vmap(critic)(replay_buffer.states).reshape(-1),
        dones=replay_buffer.dones,
        gamma=gamma,
        lambda_=lambda_,
    )
    logits = eqx.filter_vmap(policy)(replay_buffer.states)
    log_probs = jax.nn.log_softmax(logits)
    log_probs_actions = jnp.take_along_axis(
        log_probs, jnp.expand_dims(replay_buffer.actions, -1), axis=1
    )
    prob_ration_t = jnp.exp(
        log_probs_actions.reshape(-1) - replay_buffer.log_probs.reshape(-1)
    )
    return rlax.clipped_surrogate_pg_loss(
        prob_ratios_t=prob_ration_t,
        adv_t=advantages,
        epsilon=epsilon,
    )


def step_actor_network(
    actor: PyTree,
    optimiser: optax.GradientTransformation,
    optimiser_state: optax.OptState,
    critic: PyTree,
    replay_buffer: ReplayBuffer,
    epsilon: float = 0.2,
    gamma: float = 0.99,
    lambda_: float = 0.95,
) -> Tuple[PyTree, optax.OptState]:
    grad = eqx.filter_grad(objective_fn)(
        actor, critic, replay_buffer, epsilon, gamma, lambda_
    )
    updates, optimiser_state = optimiser.update(grad, optimiser_state, actor)
    actor = eqx.apply_updates(actor, updates)

    return actor, optimiser_state


def critic_loss_fn(
    critic: PyTree,
    replay_buffer: ReplayBuffer,
    epsilon: float = 0.2,
    gamma: float = 0.99,
    lambda_: float = 0.95,
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
    advantages = calculate_gae(
        rewards=replay_buffer.rewards,
        values=eqx.filter_vmap(critic)(replay_buffer.states).reshape(-1),
        dones=replay_buffer.dones,
        gamma=gamma,
        lambda_=lambda_,
    )
    v_loss_unclipped = (advantages - values) ** 2
    v_clipped = values + jnp.clip(advantages - values, -epsilon, epsilon)
    v_loss_clipped = (advantages - v_clipped) ** 2
    return jnp.mean(jnp.maximum(v_loss_unclipped, v_loss_clipped)) * 0.5
    # return jnp.mean((replay_buffer.rewards - values) ** 2)


def step_critic_network(
    critic: PyTree,
    optimiser: optax.GradientTransformation,
    optimiser_state: optax.OptState,
    replay_buffer: ReplayBuffer,
    epsilon: float = 0.2,
    gamma: float = 0.99,
    lambda_: float = 0.95,
) -> Tuple[PyTree, optax.OptState]:
    grad = eqx.filter_grad(critic_loss_fn)(
        critic, replay_buffer, epsilon, gamma, lambda_
    )
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
    epsilon: float = 0.2,
    gamma: float = 0.99,
    lambda_: float = 0.95,
) -> (PyTree, PyTree, Array):
    critic, critic_optimiser_state = step_critic_network(
        critic=critic,
        optimiser=critic_optimiser,
        optimiser_state=critic_optimiser_state,
        replay_buffer=replay_buffer,
        epsilon=epsilon,
        gamma=gamma,
        lambda_=lambda_,
    )
    actor, actor_optimiser_state = step_actor_network(
        actor=actor,
        optimiser=actor_optimiser,
        optimiser_state=actor_optimiser_state,
        critic=critic,
        replay_buffer=replay_buffer,
        epsilon=epsilon,
        gamma=gamma,
        lambda_=lambda_,
    )

    return actor, actor_optimiser_state, critic, critic_optimiser_state
