import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
from jaxtyping import Array


def get_future_rewards(rewards: Array, gamma=0.99) -> Array:
    """Calculate the future rewards for a given set of rewards.
    Args:
        rewards: The rewards to calculate the future rewards for.
        gamma: The discount factor.

    Returns:
        The future rewards.
    """
    returns = jnp.zeros_like(rewards)
    future_returns = 0

    for t in range(len(rewards) - 1, -1, -1):
        future_returns = rewards[t] + gamma * future_returns
        returns = returns.at[t].set(future_returns)

    return returns


def get_policy_gradient_discrete_loss(
    logits: jnp.ndarray,
    actions: jnp.ndarray,
    advantages: jnp.ndarray,
) -> jnp.ndarray:
    """Calculate the policy gradient loss for a discrete action space.
    Args:
        logits: The logits of the policy.
        actions: The actions taken.
        advantages: The advantages of the actions taken.
    Returns:
        The policy gradient loss.
    """
    log_probs = tfp.distributions.Categorical(logits=logits).log_prob(actions)
    advantages = jax.lax.stop_gradient(advantages)
    return -jnp.mean(log_probs * advantages)


@jax.jit
def get_discounted_rewards(rewards: jnp.ndarray, gamma=0.99) -> jnp.ndarray:
    """Calculate the discounted rewards for a given set of rewards.
    Args:
        rewards: The rewards to calculate the discounted rewards for.
        gamma: The discount factor.
    Returns:
        The discounted rewards.
    """

    def body_fn(i: int, val: float):
        return val + (gamma**i) * rewards[i]

    discounted_rewards = jnp.zeros(())
    num_rewards = len(rewards)
    discounted_rewards = jax.lax.fori_loop(0, num_rewards, body_fn, discounted_rewards)

    return discounted_rewards


@jax.jit
def get_total_discounted_rewards(rewards: jnp.array, gamma=0.99) -> jnp.array:
    """Calculate the total discounted rewards for a given set of rewards.
    Args:
        rewards: The rewards to calculate the total discounted rewards for.
        gamma: The discount factor.
    Returns:
        The total discounted rewards.
    """
    def scan_fn(carry, current_reward):
        discounted_reward = carry + current_reward
        return discounted_reward * gamma, discounted_reward

    _, total_discounted_rewards = jax.lax.scan(scan_fn, 0.0, rewards[::-1])
    return total_discounted_rewards[::-1].reshape(-1, 1)
