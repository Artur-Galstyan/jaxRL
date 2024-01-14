import random
from typing import NamedTuple

import equinox as eqx
import gymnasium as gym
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from gymnasium.wrappers.time_limit import TimeLimit
from jaxtyping import Array, Float32, PRNGKeyArray
from tqdm import tqdm

from jax_rl.commons import ReplayBuffer
from jax_rl.ppo import train
from jax_rl.rollout import rollout_gym

n_episodes = 500
batch_size = 64
gamma = 0.99
lambda_ = 0.95
epsilon = 0.2
all_rewards = []
n_epochs = 20
adam_epsilon = 1e-5
n_updates = 30


class Actor(eqx.Module):
    """Policy network for the policy gradient algorithm in a discrete action space."""

    mlp: eqx.nn.MLP

    def __init__(
        self,
        in_size: int,
        out_size: int,
        key: PRNGKeyArray,
        width_size: int = 128,
        depth: int = 3,
    ) -> None:
        key, *subkeys = jax.random.split(key, 5)
        self.mlp = eqx.nn.MLP(
            in_size=in_size,
            out_size=out_size,
            width_size=width_size,
            depth=depth,
            key=key,
            activation=jax.nn.tanh,
        )

    def __call__(self, x: Float32[Array, "state_dims"]) -> Array:
        return self.mlp(x)


class Critic(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(
        self,
        in_size: int,
        out_size: int,
        key: PRNGKeyArray,
        width_size: int = 128,
        depth: int = 3,
    ) -> None:
        key, *subkeys = jax.random.split(key, 5)
        self.mlp = eqx.nn.MLP(
            in_size=in_size,
            out_size=out_size,
            width_size=width_size,
            depth=depth,
            key=key,
            activation=jax.nn.tanh,
        )

    def __call__(self, x: Float32[Array, "state_dims"]) -> Array:
        return self.mlp(x)


max_episode_steps = 1024
# env_name = "CartPole-v1"
env_name = "LunarLander-v2"
# env_name = "Acrobot-v1"
env = gym.make(env_name)
env = TimeLimit(env, max_episode_steps=max_episode_steps - 1)
learning_rate = 3e-4


assert env.observation_space.shape is not None, "Observation space must be defined."

actor = Actor(
    in_size=env.observation_space.shape[0],
    out_size=env.action_space.n,
    key=jax.random.PRNGKey(0),
)

critic = Critic(
    in_size=env.observation_space.shape[0],
    out_size=1,
    key=jax.random.PRNGKey(1),
)

scheduler = optax.linear_schedule(
    init_value=learning_rate,
    end_value=0,
    transition_steps=n_episodes * n_epochs,
)
actor_optimiser = optax.adam(scheduler, eps=1e-5)
critic_optimiser = optax.adam(scheduler, eps=1e-5)
critic_optimiser_state = critic_optimiser.init(eqx.filter(critic, eqx.is_inexact_array))
actor_optimiser_state = actor_optimiser.init(eqx.filter(actor, eqx.is_inexact_array))
key = jax.random.PRNGKey(2)

training_stats_log = tqdm(
    total=float("-inf"), position=2, leave=False, ncols=0, ascii=True
)

for epoch in tqdm(range(n_epochs), desc="Epochs", position=0, leave=False, ascii=True):
    replay_buffers: list[ReplayBuffer] = []
    rews = []
    for eps in tqdm(
        range(n_episodes), desc="Episodes", position=1, leave=False, ascii=True
    ):
        key, subkey = jax.random.split(key)
        obs, actions, rewards, log_probs, dones = rollout_gym(env, actor, subkey)
        # pad arrays to make them all the same length of max_episode_steps
        obs = np.pad(obs, ((0, max_episode_steps - len(obs)), (0, 0)))
        actions = np.pad(actions, (0, max_episode_steps - len(actions)))
        rewards = np.pad(rewards, (0, max_episode_steps - len(rewards) - 1))
        log_probs = np.pad(log_probs, (0, max_episode_steps - len(log_probs)))
        dones = np.pad(dones, (True, max_episode_steps - len(dones)))

        rews.append(jnp.sum(rewards))

        b = ReplayBuffer(
            states=jnp.array(obs),
            actions=jnp.array(actions),
            rewards=jnp.array(rewards),
            log_probs=jnp.array(log_probs),
            dones=jnp.array(dones),
        )
        replay_buffers.append(b)
    # shuffle the replay buffers
    random.shuffle(replay_buffers)
    for update in tqdm(
        range(n_updates), desc="Updates", position=4, leave=False, ascii=True
    ):
        for b in replay_buffers:
            actor, actor_optimiser_state, critic, critic_optimiser_state = train(
                actor=actor,
                actor_optimiser=actor_optimiser,
                critic=critic,
                critic_optimiser=critic_optimiser,
                actor_optimiser_state=actor_optimiser_state,
                critic_optimiser_state=critic_optimiser_state,
                gamma=gamma,
                lambda_=lambda_,
                epsilon=epsilon,
                replay_buffer=b,
                max_episode_steps=max_episode_steps,
            )
    all_rewards.append(jnp.mean(jnp.array(rews)))
    training_stats_log.set_description(
        f" Mean reward: {float(np.mean(np.array(all_rewards))):.2f}",
    )

plt.plot(all_rewards)
plt.show()
