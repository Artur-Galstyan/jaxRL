from functools import partial

import equinox as eqx
import gymnasium
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
import rlax
from jaxonloader import DataLoader, StandardDataset
from jaxtyping import Array, Float, PRNGKeyArray, PyTree
from tqdm import tqdm

from jax_rl.commons import ReplayBuffer
from jax_rl.rollout import rollout_gym


batch_size = 8
gamma = 0.99
lambda_ = 0.95
clip_ratio = 0.2
learning_rate = 3e-4
env_name = "LunarLander-v2"
env = gymnasium.make(env_name)
n_actions = env.action_space.n  # type: ignore
assert env.observation_space.shape is not None, "Observation space must be defined."
n_dims = env.observation_space.shape[0]
n_episodes = 500


class Agent(eqx.Module):
    actor: eqx.nn.MLP
    critic: eqx.nn.MLP

    def __init__(self, in_size: int, out_size: int, *, key: PRNGKeyArray) -> None:
        key, *subkeys = jax.random.split(key, 3)
        self.actor = eqx.nn.MLP(
            in_size, out_size, width_size=32, depth=3, key=subkeys[0]
        )
        self.critic = eqx.nn.MLP(in_size, 1, width_size=32, depth=3, key=subkeys[1])

    def __call__(self, x: Float[Array, " n_dims"]) -> tuple[Array, Array]:
        return self.actor(x), self.critic(x)


def objective_fn(
    agent: Agent,
    replay_buffer: ReplayBuffer,
    clip_ratio: float,
    gamma: float,
    lambda_: float,
) -> Float[Array, ""]:
    actor_logits, values = eqx.filter_vmap(agent)(replay_buffer.states)
    log_probs = jax.nn.log_softmax(actor_logits)
    log_probs_actions = jnp.take_along_axis(
        log_probs, jnp.expand_dims(replay_buffer.actions, -1), axis=1
    )

    advantages = rlax.truncated_generalized_advantage_estimation(
        r_t=replay_buffer.rewards.reshape(-1)[:-1],
        discount_t=jnp.array([gamma] * (len(replay_buffer.rewards) - 1)).reshape(-1),
        values=values.reshape(-1),
        lambda_=lambda_,
        stop_target_gradients=True,
    )

    ratio = jnp.exp(log_probs_actions.reshape(-1) - replay_buffer.log_probs.reshape(-1))
    advantages = jnp.append(advantages, jnp.array(0.0))
    loss = rlax.clipped_surrogate_pg_loss(
        prob_ratios_t=ratio,
        adv_t=advantages,
        epsilon=clip_ratio,
    )

    return jnp.array(loss)


@eqx.filter_jit
def step(
    agent: PyTree,
    optimiser: optax.GradientTransformation,
    optimiser_state: optax.OptState,
    replay_buffer: ReplayBuffer,
):
    grad = eqx.filter_grad(objective_fn)(
        agent, replay_buffer, clip_ratio, gamma, lambda_
    )
    updates, optimiser_state = optimiser.update(grad, optimiser_state, agent)
    agent = eqx.apply_updates(agent, updates)

    return agent, optimiser_state


agent = Agent(n_dims, n_actions, key=jax.random.PRNGKey(0))
agent_optimiser = optax.adam(learning_rate)
agent_optimiser_state = agent_optimiser.init(eqx.filter(agent, eqx.is_inexact_array))
key = jax.random.PRNGKey(2)

all_rewards = []

for eps in tqdm(range(n_episodes)):
    key, subkey, subkey2 = jax.random.split(key, 3)
    partial_agent = lambda x: agent(x)[0]
    dataset = rollout_gym(env, partial_agent, subkey)

    dataloader = DataLoader(
        batch_size=batch_size, shuffle=False, drop_last=True, dataset=dataset
    )

    all_rewards.append(jnp.sum(dataset.columns[2]))

    for batch in dataloader:
        obs, actions, rewards, log_probs, dones = batch
        b = ReplayBuffer(
            states=jnp.array(obs),
            actions=jnp.array(actions),
            rewards=jnp.array(rewards, dtype=jnp.float32),
            log_probs=jnp.array(log_probs),
            dones=jnp.array(dones),
        )

        agent, agent_optimiser_state = step(
            agent, agent_optimiser, agent_optimiser_state, b
        )

plt.plot(all_rewards)
plt.show()
