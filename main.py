import gymnasium
import jax
import optax

from jax_rl import policy_gradient, ppo


def main():
    env = gymnasium.make("CartPole-v1", max_episode_steps=200, render_mode="rgb_array")

    actor_optimiser = optax.adam(learning_rate=1e-3)
    critic_optimiser = optax.adam(learning_rate=1e-3)
    key = jax.random.PRNGKey(0)
    actor = ppo.train(
        env,
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
