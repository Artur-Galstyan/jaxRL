from typing import Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import rlax
from jaxtyping import Array, PRNGKeyArray, PyTree
from jax_rl.commons import ReplayBuffer
import gymnasium as gym


n_envs = 4
n_steps = 200
seed = 42
capture_video = False


class Actor(eqx.Module):
    def __init__(self, envs: gym.vector.SyncVectorEnv):
        pass


class Critic(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(
        self,
        envs: gym.vector.SyncVectorEnv,
        width: int = 64,
        depth: int = 2,
        *,
        key: PRNGKeyArray,
    ):
        in_size = jnp.array(envs.single_observation_space.shape).prod()
        out_size = 1
        self.mlp = eqx.nn.MLP(
            in_size=in_size,
            out_size=out_size,
            width=width,
            depth=depth,
            key=key,
            activation=jax.nn.tanh,
        )


def make_env(gym_id: str, idx: int, capture_video: bool = False) -> gym.Env:
    def thunk() -> gym.Env:
        env = gym.make(gym_id, render_mode="rgb_array")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(
                env,
                video_folder="videos",
                name_prefix=gym_id,
                episode_trigger=lambda x: x % 100 == 0 and x > 0,
            )
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


envs = gym.vector.SyncVectorEnv(
    [make_env("CartPole-v1", i, capture_video) for i in range(n_envs)]
)
obs, info = envs.reset(seed=seed)
for _ in range(n_steps):
    action = envs.action_space.sample()
    obs, reward, terminated, truncated, info = envs.step(action)
    if "final_info" in info:
        final_infos = info["final_info"]
        for final_info in final_infos:
            if final_info is not None:
                print(final_info["episode"]["r"])
