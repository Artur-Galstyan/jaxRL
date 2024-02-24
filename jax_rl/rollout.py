import equinox as eqx
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from jaxonloader import StandardDataset
from jaxtyping import PRNGKeyArray, PyTree


def rollout_gym(
    env: gym.Env,
    policy: PyTree,
    key: PRNGKeyArray,
    render: bool = False,
) -> StandardDataset:
    """Rollout a policy in a gym environment."""

    obs, info = env.reset()

    observations = [obs]
    actions = []
    log_probs = []
    rewards = []
    dones = []
    info = None

    policy = eqx.filter_jit(policy)

    while True:
        key, subkey = jax.random.split(key)
        logits = policy(obs)
        action = jax.random.categorical(logits=logits, key=subkey)
        action = np.array(action)
        log_prob = np.array(jax.nn.log_softmax(np.array(logits)))
        log_probs.append(log_prob[action])
        obs, reward, terminated, truncated, info = env.step(action)
        if render:
            env.render()

        observations.append(obs)
        done = terminated or truncated
        actions.append(action)
        rewards.append(reward)
        dones.append(done)

        if done:
            break
    return StandardDataset(
        jnp.array(observations, dtype=jnp.float32),
        jnp.array(actions),
        jnp.array(rewards),
        jnp.array(log_probs),
        jnp.array(dones),
    )
