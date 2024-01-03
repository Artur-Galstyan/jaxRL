import gymnasium as gym
import jax
import numpy as np
from jaxtyping import PRNGKeyArray, PyTree
from jax_rl.commons import RLDataset


def rollout_gym(
    env: gym.Env,
    policy: PyTree,
    key: PRNGKeyArray,
    render: bool = False,
):
    """Rollout a policy in a gym environment."""

    obs, info = env.reset()

    observations = [obs]
    actions = []
    log_probs = []
    rewards = []
    dones = []
    info = None

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

    return (
        np.array(observations),
        np.array(actions),
        np.array(rewards),
        np.array(log_probs),
        np.array(dones),
    )

    return dataset
