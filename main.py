import jax
import jax.numpy as jnp
import optax
import gymnasium

from equinox_rl import policy_gradient
from equinox_rl.policy_gradient import Policy


def main():
    env = gymnasium.make("CartPole-v1", max_episode_steps=200, render_mode="rgb_array")
    env = gymnasium.wrappers.RecordEpisodeStatistics(env)
    env = gymnasium.wrappers.RecordVideo(
        env,
        "videos/",
        name_prefix="jax-cartpole",
        episode_trigger=lambda x: x % 1000 == 0,
    )
    key = jax.random.PRNGKey(0)
    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n

    key, subkey = jax.random.split(key)
    policy = Policy(in_size=obs_dim, out_size=n_acts, key=subkey)

    optimizer = optax.adam(learning_rate=1e-3)

    policy = policy_gradient.train(
        policy,
        env,
        optimizer,
        n_epochs=30,
        n_episodes=1000,
    )


if __name__ == "__main__":
    main()
