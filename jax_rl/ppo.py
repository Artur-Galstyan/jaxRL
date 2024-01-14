from typing import Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import rlax
from jaxtyping import Array, PyTree
from jax_rl.commons import ReplayBuffer


def create_mask(episode_length, max_eps_length):
    mask = jnp.zeros(shape=(max_eps_length))
    mask = jnp.where(jnp.arange(max_eps_length) < episode_length, 1.0, 0.0)
    return mask


def normalize_advantages(advantages: Array) -> Array:
    return (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)


def objective_fn(
    policy: PyTree,
    critic: PyTree,
    replay_buffer: ReplayBuffer,
    epsilon: float = 0.2,
    gamma: float = 0.99,
    lambda_: float = 0.95,
    max_episode_steps: int = 500,
) -> Array:
    mask = create_mask(len(replay_buffer.states), max_episode_steps)
    advantages = rlax.truncated_generalized_advantage_estimation(
        r_t=replay_buffer.rewards,
        discount_t=jnp.array([gamma] * (len(replay_buffer.rewards))),
        values=eqx.filter_vmap(critic)(replay_buffer.states).reshape(-1),
        lambda_=lambda_,
        # stop_target_gradients=True,
    )

    advantages = jnp.append(advantages, 0.0)
    advantages = advantages * mask
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
    max_episode_steps: int = 500,
) -> Tuple[PyTree, optax.OptState]:
    grad = eqx.filter_grad(objective_fn)(
        actor, critic, replay_buffer, epsilon, gamma, lambda_, max_episode_steps
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
    max_episode_steps: int = 500,
) -> Array:
    mask = create_mask(len(replay_buffer.states), max_episode_steps)

    advantages = rlax.truncated_generalized_advantage_estimation(
        r_t=replay_buffer.rewards,
        discount_t=jnp.array([gamma] * (len(replay_buffer.rewards))),
        values=eqx.filter_vmap(critic)(replay_buffer.states).reshape(-1),
        lambda_=lambda_,
        # stop_target_gradients=True,
    )

    advantages = jnp.append(advantages, 0.0) * mask
    values = eqx.filter_vmap(critic)(replay_buffer.states)
    v_loss_unclipped = (advantages - values) ** 2
    v_clipped = values + jnp.clip(advantages - values, -epsilon, epsilon)
    v_loss_clipped = (advantages - v_clipped) ** 2
    return jnp.mean((jnp.maximum(v_loss_unclipped, v_loss_clipped)) * 0.5)


def step_critic_network(
    critic: PyTree,
    optimiser: optax.GradientTransformation,
    optimiser_state: optax.OptState,
    replay_buffer: ReplayBuffer,
    epsilon: float = 0.2,
    gamma: float = 0.99,
    lambda_: float = 0.95,
    max_episode_steps: int = 500,
) -> Tuple[PyTree, optax.OptState]:
    grad = eqx.filter_grad(critic_loss_fn)(
        critic, replay_buffer, epsilon, gamma, lambda_, max_episode_steps
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
    max_episode_steps: int = 500,
) -> (PyTree, PyTree, PyTree, PyTree):
    critic, critic_optimiser_state = step_critic_network(
        critic=critic,
        optimiser=critic_optimiser,
        optimiser_state=critic_optimiser_state,
        replay_buffer=replay_buffer,
        epsilon=epsilon,
        gamma=gamma,
        lambda_=lambda_,
        max_episode_steps=max_episode_steps,
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
        max_episode_steps=max_episode_steps,
    )

    return actor, actor_optimiser_state, critic, critic_optimiser_state
