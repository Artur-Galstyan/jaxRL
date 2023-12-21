import gymnasium
import jax
import optax

from equinox_rl import policy_gradient


def main():
    env = gymnasium.make("CartPole-v1", max_episode_steps=200, render_mode="rgb_array")

    optimizer = optax.adam(learning_rate=1e-3)
    key = jax.random.PRNGKey(0)
    policy = policy_gradient.train(
        env,
        optimizer,
        n_epochs=30,
        n_episodes=1000,
        key=key,
    )


if __name__ == "__main__":
    main()
