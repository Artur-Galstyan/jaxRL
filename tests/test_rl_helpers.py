import jax.numpy as jnp
import jax
import numpy as np

from jax_rl.common.rl_helpers import get_policy_gradient_discrete_loss
from jax_rl.common.rl_helpers import calculate_gae


def test_get_policy_gradient_discrete_loss():
    # Test case 1: Basic test
    logits = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    actions = jnp.array([0, 1])
    advantages = jnp.array([1.0, 2.0])

    loss = get_policy_gradient_discrete_loss(logits, actions, advantages)

    softmax = jnp.exp(logits) / jnp.sum(jnp.exp(logits), axis=1, keepdims=True)
    log_probs = jnp.log(softmax[jnp.arange(logits.shape[0]), actions])
    expected_loss = -jnp.mean(log_probs * advantages)

    assert np.isclose(loss, expected_loss), f"Expected {expected_loss}, but got {loss}"

    # Test case 2: Check if stop_gradient is working
    policy_gradient_loss_grad = jax.grad(get_policy_gradient_discrete_loss, argnums=2)
    grad_advantages = policy_gradient_loss_grad(logits, actions, advantages)
    assert jnp.all(
        grad_advantages == 0
    ), f"Expected all zeros, but got {grad_advantages}"

    # Test case 3: Check when loss is zero
    logits = jnp.array([[1.0, 1.0], [1.0, 1.0]])
    actions = jnp.array([0, 1])
    advantages = jnp.array([0.0, 0.0])

    loss = get_policy_gradient_discrete_loss(logits, actions, advantages)
    expected_loss = 0.0
    assert np.isclose(loss, expected_loss), f"Expected {expected_loss}, but got {loss}"


def test_gae_calculation():
    rewards = jnp.array([1, 2, 3], dtype=jnp.float32)
    values = jnp.array([1, 2, 3], dtype=jnp.float32)
    dones = jnp.array([0, 0, 0], dtype=jnp.float32)
    gamma = 0.99
    lam = 0.95
    advantages = calculate_gae(rewards, values, dones, gamma, lam)
    expected_advantages = jnp.array(
        [2.64665, 1.4705882, 1.0], dtype=jnp.float32
    )  # Adjust based on your implementation
    jnp.allclose(advantages, expected_advantages, rtol=1e-5)
