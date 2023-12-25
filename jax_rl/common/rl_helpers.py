import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def get_policy_gradient_discrete_loss(
    logits: Array,
    actions: Array,
    advantages: Array,
) -> Array:
    """Calculate the policy gradient loss for a discrete action space.
    Args:
        logits: The logits of the policy.
        actions: The actions taken.
        advantages: The advantages of the actions taken.
    Returns:
        The policy gradient loss.
    """
    log_probs = jax.nn.log_softmax(logits)
    log_probs = jnp.take_along_axis(log_probs, jnp.expand_dims(actions, -1), axis=1)
    advantages = jax.lax.stop_gradient(advantages)
    return -jnp.mean(log_probs * advantages)


def get_discounted_rewards(rewards: Array, gamma=0.99) -> Float[Array, ""]:
    """Calculate the discounted rewards for a given set of rewards.
    Args:
        rewards: The rewards to calculate the discounted rewards for.
        gamma: The discount factor.
    Returns:
        The discounted rewards for the given rewards as a 1D array (i.e. a scalar).
    """

    def body_fn(i: int, val: float):
        return val + (gamma**i) * rewards[i]

    discounted_rewards = jnp.zeros(())
    num_rewards = len(rewards)
    discounted_rewards = jax.lax.fori_loop(0, num_rewards, body_fn, discounted_rewards)

    return discounted_rewards


def get_total_discounted_rewards(rewards: Float[Array, "n_steps"], gamma=0.99) -> Array:
    """Calculate the total discounted rewards for a given set of rewards.
    This is also known as the rewards-to-go.

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

    total_discounted_rewards = total_discounted_rewards[::-1].reshape(
        -1,
    )
    assert (
        total_discounted_rewards.shape == rewards.shape
    ), f"total_discounted_rewards.shape: {total_discounted_rewards.shape}, rewards.shape: {rewards.shape}"

    return total_discounted_rewards


def calculate_gae(
    rewards: Array,
    values: Array,
    dones: Array,
    gamma: float,
    lam: float,
) -> Array:
    def body_fun(
        carry: tuple[Array, Array], t: Array
    ) -> tuple[tuple[Array, Array], None]:
        advantages, gae_inner = carry
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        gae_inner = delta + gamma * lam * (1 - dones[t]) * gae_inner
        advantages = advantages.at[t].set(gae_inner)
        return (advantages, gae_inner), None

    values = jnp.append(values, values[0])
    advt = jnp.zeros_like(rewards)
    gae = jnp.array(0.0)
    t = len(rewards)

    (advt, _), _ = jax.lax.scan(body_fun, (advt, gae), jnp.arange(t - 1, -1, -1))
    return advt


if __name__ == "__main__":
    from icecream import ic

    r = jnp.array([1 for _ in range(10)])
    ic(get_total_discounted_rewards(r))
