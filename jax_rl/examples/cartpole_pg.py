import equinox as eqx
import gymnasium as gym
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from jaxtyping import Array, Float32, PRNGKeyArray
from torch.utils.data import DataLoader
from tqdm import tqdm

from jax_rl.commons import ReplayBuffer
from jax_rl.policy_gradient import train
from jax_rl.rollout import rollout_gym


class Actor(eqx.Module):
    """Policy network for the policy gradient algorithm in a discrete action space."""

    mlp: eqx.nn.MLP

    def __init__(
        self,
        in_size: int,
        out_size: int,
        key: PRNGKeyArray,
        width_size: int = 32,
        depth: int = 2,
    ) -> None:
        key, *subkeys = jax.random.split(key, 5)
        self.mlp = eqx.nn.MLP(
            in_size=in_size,
            out_size=out_size,
            width_size=width_size,
            depth=depth,
            key=key,
        )

    def __call__(self, x: Float32[Array, "state_dims"]) -> Array:
        """Forward pass of the policy network.
        Args:
            x: The input to the policy network.
        Returns:
            The output of the policy network.
        """
        return self.mlp(x)


class Critic(eqx.Module):
    """Value network for the policy gradient algorithm in a discrete action space."""

    mlp: eqx.nn.MLP

    def __init__(
        self,
        in_size: int,
        out_size: int,
        key: PRNGKeyArray,
        width_size: int = 64,
        depth: int = 3,
    ) -> None:
        key, *subkeys = jax.random.split(key, 5)
        self.mlp = eqx.nn.MLP(
            in_size=in_size,
            out_size=out_size,
            width_size=width_size,
            depth=depth,
            key=key,
        )

    def __call__(self, x: Float32[Array, "state_dims"]) -> Array:
        """Forward pass of the policy network.
        Args:
            x: The input to the policy network.
        Returns:
            The output of the policy network.
        """
        return self.mlp(x)


env = gym.make("CartPole-v1")

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

actor_optimiser = optax.adam(learning_rate)
critic_optimiser = optax.adam(learning_rate)
critic_optimiser_state = critic_optimiser.init(eqx.filter(critic, eqx.is_inexact_array))
actor_optimiser_state = actor_optimiser.init(eqx.filter(actor, eqx.is_inexact_array))
key = jax.random.PRNGKey(2)

n_episodes = 10000
batch_size = 8

all_rewards = []
for eps in tqdm(range(n_episodes)):
    key, subkey, subkey2 = jax.random.split(key, 3)
    dataset = rollout_gym(env, actor, subkey)

    dataloader = DataLoader(
        batch_size=batch_size, shuffle=False, drop_last=True, dataset=dataset
    )

    all_rewards.append(jnp.sum(dataset.rewards.numpy()))

    for batch in dataloader:
        obs, actions, rewards, log_probs, dones = batch
        b = ReplayBuffer(
            states=jnp.array(obs.numpy()),
            actions=jnp.array(actions.numpy()),
            rewards=jnp.array(rewards.numpy()),
            log_probs=jnp.array(log_probs.numpy()),
            dones=jnp.array(dones.numpy()),
        )

        actor, actor_optimiser_state, critic, critic_optimiser_state = train(
            actor=actor,
            actor_optimiser=actor_optimiser,
            actor_optimiser_state=actor_optimiser_state,
            critic=critic,
            critic_optimiser=critic_optimiser,
            critic_optimiser_state=critic_optimiser_state,
            replay_buffer=b,
            key=subkey2,
        )

plt.plot(all_rewards)
plt.show()
