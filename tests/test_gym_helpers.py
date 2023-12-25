from jax_rl.common.gym_helpers import rollout_discrete
import gymnasium as gym


def test_rollout_discrete():
    env = gym.make("CartPole-v1")

    def action_fn(obs, **kwargs):
        return env.action_space.sample()

    observations, actions, rewards, dones = rollout_discrete(env, action_fn, {})

    assert observations.shape[1] == 4
    assert actions.shape == rewards.shape == dones.shape

    env.close()
