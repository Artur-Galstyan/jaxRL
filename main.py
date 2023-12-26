import jax
import optax
import gymnax

from jax_rl import policy_gradient, ppo


def main():
    # env = gymnasium.make(
    #     "LunarLander-v2", max_episode_steps=200, render_mode="rgb_array"
    # )

    env, env_params = gymnax.make("Acrobot-v1")
    obs_size = env.observation_space(env_params).shape[0]
    action_size = env.action_space(env_params).n
    actor_optimiser = optax.adam(learning_rate=1e-3)
    critic_optimiser = optax.adam(learning_rate=1e-3)
    key = jax.random.PRNGKey(0)
    actor = ppo.train(
        env,
        env_type="gymnax",
        env_params=env_params,
        obs_size=obs_size,
        action_size=action_size,
        actor=None,
        critic=None,
        actor_optimiser=actor_optimiser,
        critic_optimiser=critic_optimiser,
        n_epochs=30,
        n_episodes=1000,
        key=key,
    )


if __name__ == "__main__":
    main()
