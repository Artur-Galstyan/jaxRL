import sys
import time
import matplotlib.pyplot as plt
import equinox as eqx
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxtyping import Array, PRNGKeyArray
import rlax
from tqdm import tqdm

gym_id = "CartPole-v1"
learning_rate = 3e-4
clip_vloss = True
vf_coef = 0.5
max_grad_norm = 0.5
target_kl = None
approx_kl = 0.01
seed = 0
n_total_steps = 25000
n_envs = 8
n_steps = 128
anneal_lr = True
gae = True
gamma = 0.99
lambda_ = 0.95
n_minibatches = 4
n_update_epochs = 4
normalize_advantages = True
epsilon = 0.2
entropy_coef = 0.1

batch_size = n_envs * n_steps
minibatch_size = batch_size // n_minibatches
n_updates = n_total_steps // batch_size

scheduler = (
    optax.linear_schedule(
        init_value=learning_rate, end_value=0, transition_steps=n_updates
    )
    if anneal_lr
    else learning_rate
)


def make_env(gym_id, seed):
    def thunk():
        env = gym.make(gym_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


envs = gym.vector.SyncVectorEnv([make_env(gym_id, seed + i) for i in range(n_envs)])


class PPO(eqx.Module):
    critic: eqx.nn.Sequential
    actor: eqx.nn.Sequential

    def __init__(self, envs: gym.vector.SyncVectorEnv, key: PRNGKeyArray):
        key, *subkeys = jax.random.split(key, 7)
        self.critic = eqx.nn.Sequential(
            [
                eqx.nn.Linear(
                    jnp.array(envs.single_observation_space.shape).prod(),
                    64,
                    key=subkeys[0],
                ),
                eqx.nn.Lambda(jax.nn.tanh),
                eqx.nn.Linear(64, 64, key=subkeys[1]),
                eqx.nn.Lambda(jax.nn.tanh),
                eqx.nn.Linear(64, "scalar", key=subkeys[2]),
            ]
        )
        self.actor = eqx.nn.Sequential(
            [
                eqx.nn.Linear(
                    jnp.array(envs.single_observation_space.shape).prod(),
                    64,
                    key=subkeys[3],
                ),
                eqx.nn.Lambda(jax.nn.tanh),
                eqx.nn.Linear(64, 64, key=subkeys[4]),
                eqx.nn.Lambda(jax.nn.tanh),
                eqx.nn.Linear(64, envs.single_action_space.n, key=subkeys[5]),
            ]
        )

    def __call__(
        self, obs: Array, key: PRNGKeyArray, action: Array = None
    ) -> (Array, Array, Array, Array):
        logits = self.actor(obs)
        log_probs = jax.nn.log_softmax(logits)
        if action is None:
            action = jax.random.categorical(key, logits=logits)
        action_log_prob = jnp.take_along_axis(
            log_probs, jnp.expand_dims(action, axis=-1), axis=-1
        )
        value = self.critic(obs)
        entropy = -jnp.sum(log_probs * jnp.exp(log_probs), axis=-1)
        return action, action_log_prob, value, entropy


@eqx.filter_jit
def get_value(ppo, obs: Array):
    return eqx.filter_vmap(ppo.critic)(obs)


key = jax.random.PRNGKey(seed)
key, subkey = jax.random.split(key)
ppo = PPO(envs, subkey)

optimizer = optax.adamw(scheduler, eps=1e-5)
opt_state = optimizer.init(eqx.filter(ppo, eqx.is_inexact_array_like))

obs = jnp.zeros(shape=(n_envs, n_steps) + envs.single_observation_space.shape)
actions = jnp.zeros((n_envs, n_steps) + envs.single_action_space.shape)
logprobs = jnp.zeros(shape=(n_envs, n_steps))
rewards = jnp.zeros(shape=(n_envs, n_steps))
dones = jnp.zeros(shape=(n_envs, n_steps))
values = jnp.zeros(shape=(n_envs, n_steps))

# ppo = eqx.filter_jit(ppo)


@jax.jit
def update_buffers(
    obs: Array,
    actions: Array,
    logprobs: Array,
    rewards: Array,
    dones: Array,
    values: Array,
    curr_obs: Array,
    curr_actions: Array,
    curr_logprobs: Array,
    curr_rewards: Array,
    curr_dones: Array,
    curr_values: Array,
    step: int,
):
    obs = obs.at[:, step].set(curr_obs)
    actions = actions.at[:, step].set(curr_actions)
    logprobs = logprobs.at[:, step].set(curr_logprobs)
    rewards = rewards.at[:, step].set(curr_rewards)
    dones = dones.at[:, step].set(curr_dones)
    values = values.at[:, step].set(curr_values)
    return obs, actions, logprobs, rewards, dones, values


@eqx.filter_jit
def step_fn(
    ppo,
    opt_state,
    optimizer,
    b_obs,
    b_actions,
    b_logprobs,
    b_advantages,
    b_returns,
    b_values,
    mb_inds,
    normalize_advantages,
    clip_vloss,
    epsilon,
    key,
):
    grads = eqx.filter_grad(ppo_loss)(
        ppo,
        b_obs,
        b_actions,
        b_logprobs,
        b_advantages,
        b_returns,
        b_values,
        mb_inds,
        normalize_advantages,
        clip_vloss,
        epsilon,
        key,
    )
    updates, opt_state = optimizer.update(grads, opt_state, ppo)
    ppo = eqx.apply_updates(ppo, updates)
    return ppo, opt_state


def ppo_loss(
    ppo,
    b_obs,
    b_actions,
    b_logprobs,
    b_advantages,
    b_returns,
    b_values,
    mb_inds,
    normalize_advantages,
    clip_vloss,
    epsilon,
    key,
):
    actions = jnp.array(b_actions[mb_inds], dtype=jnp.int32)
    _, newlogprob, newvalue, entropy = eqx.filter_vmap(ppo, in_axes=(0, None, 0))(
        b_obs[mb_inds], key, actions
    )

    newlogprob = newlogprob.reshape(-1)
    logratio = newlogprob - b_logprobs[mb_inds]
    ratio = jnp.exp(logratio)

    mb_advantages = b_advantages[mb_inds]
    if normalize_advantages:
        mb_advantages = (mb_advantages - mb_advantages.mean()) / (
            mb_advantages.std() + 1e-8
        )
    # pg_loss = rlax.clipped_surrogate_pg_loss(ratio, mb_advantages, epsilon, False)

    # Policy loss
    pg_loss1 = -mb_advantages * ratio
    pg_loss2 = -mb_advantages * jnp.clip(ratio, 1 - epsilon, 1 + epsilon)
    pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()

    # Value loss
    newvalue = newvalue.reshape(-1)
    if clip_vloss:
        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
        v_clipped = b_values[mb_inds] + jnp.clip(
            newvalue - b_values[mb_inds],
            -epsilon,
            epsilon,
        )
        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
        v_loss_max = jnp.maximum(v_loss_unclipped, v_loss_clipped)
        v_loss = 0.5 * v_loss_max.mean()
    else:
        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

    entropy_loss = entropy.mean()
    loss = pg_loss - entropy_coef * entropy_loss + v_loss * vf_coef
    return loss


global_step = 0

start_time = time.time()

next_obs, info = envs.reset(seed=seed)
next_done = jnp.zeros(n_envs)

for update in range(1, n_updates + 1):
    for step in range(n_steps):
        key, subkey = jax.random.split(key)
        global_step += n_envs

        action, action_log_prob, value, entropy = eqx.filter_vmap(
            ppo, in_axes=(0, None), out_axes=0
        )(next_obs, subkey)

        next_obs_temp, n_rewards, terminated, truncated, info = envs.step(
            np.array(action)
        )
        next_done = terminated | truncated

        obs, actions, logprobs, rewards, dones, values = update_buffers(
            obs,
            actions,
            logprobs,
            rewards,
            dones,
            values,
            next_obs,
            action,
            action_log_prob.reshape(-1),
            n_rewards,
            next_done,
            value.reshape(-1),
            step,
        )
        next_obs = next_obs_temp

        for i, item in enumerate(info):
            if item == "final_info":
                eps_info = info[item][3]
                if eps_info is not None:
                    print(
                        f"global_step={global_step}, episodic_return={eps_info['episode']['r']}"
                    )
    next_value = get_value(ppo, next_obs).reshape(1, -1)
    if gae:
        advantages = jnp.zeros_like(rewards)
        lastgaelam = 0
        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[:, t + 1]
                nextvalues = values[:, t + 1]
            delta = rewards[:, t] + gamma * nextvalues * nextnonterminal - values[:, t]
            temp = delta + gamma * lambda_ * nextnonterminal * lastgaelam
            advantages = advantages.at[:, t].set(temp.reshape(-1))
            lastgaelam = advantages[:, t]
        returns = advantages + values
    else:
        returns = jnp.zeros_like(rewards)
        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                nextnonterminal = 1.0 - next_done
                next_return = next_value
            else:
                nextnonterminal = 1.0 - dones[:, t + 1]
                next_return = returns[:, t + 1]
            returns = returns.at[:, t].set(
                (rewards[:, t] + gamma * nextnonterminal * next_return).reshape(-1)
            )
    # advantages = jax.lax.stop_gradient(returns - values)
    advantages = returns - values
    b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
    b_logprobs = logprobs.reshape(-1)
    b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
    b_advantages = advantages.reshape(-1)
    b_returns = returns.reshape(-1)
    b_values = values.reshape(-1)

    b_inds = np.arange(batch_size)

    for epoch in range(n_update_epochs):
        np.random.shuffle(b_inds)
        for start in range(0, batch_size, minibatch_size):
            key, subkey = jax.random.split(key)
            end = start + minibatch_size
            mb_inds = b_inds[start:end]

            ppo, opt_state = step_fn(
                ppo,
                opt_state,
                optimizer,
                b_obs,
                b_actions,
                b_logprobs,
                b_advantages,
                b_returns,
                b_values,
                mb_inds,
                normalize_advantages,
                clip_vloss,
                epsilon,
                subkey,
            )


end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")
