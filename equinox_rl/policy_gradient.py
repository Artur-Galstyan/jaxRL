import equinox as eqx
import gymnasium
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float32, PRNGKeyArray, PyTree
from torch.utils.data import DataLoader
from tqdm import tqdm
from equinox_rl.common import gym_helpers, rl_helpers
import matplotlib.pyplot as plt


class Policy(eqx.Module):
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


@eqx.filter_jit
def get_action(
    state: Float32[Array, "state_dims"],
    policy: Policy,
    key: PRNGKeyArray,
) -> Array:
    """Get an action from the policy network.
    Args:
        state: The state to get an action from.
        policy: The policy network.
        key: The key to use for sampling.
    Returns:
        The action sampled from the policy network.
    """
    logits = policy(state)
    action = jax.random.categorical(key, logits)
    return action


def objective_fn(
    policy: PyTree,
    states: Float32[Array, "n_steps state_dim"],
    actions: Float32[Array, "n_steps"],
    rewards: Float32[Array, "n_steps"],
):
    logits = eqx.filter_vmap(policy)(states)
    log_probs = jax.nn.log_softmax(logits)
    log_probs_actions = jnp.take_along_axis(
        log_probs, jnp.expand_dims(actions, -1), axis=1
    )
    rewards = rl_helpers.get_total_discounted_rewards(
        rewards
    )  # don't let the past distract you!
    return -jnp.mean(log_probs_actions * rewards)


@eqx.filter_jit
def step(
    policy: PyTree,
    states: Float32[Array, "n_steps state_dim"],
    actions: Float32[Array, "n_steps"],
    rewards: Float32[Array, "n_steps"],
    optimiser: optax.GradientTransformation,
    optimiser_state: optax.OptState,
):
    value, grad = eqx.filter_value_and_grad(objective_fn)(
        policy, states, actions, rewards
    )
    updates, optimiser_state = optimiser.update(grad, optimiser_state, policy)
    policy = eqx.apply_updates(policy, updates)

    return policy, optimiser_state


def train(
    policy: PyTree,
    env: gymnasium.Env,
    optimiser: optax.GradientTransformation,
    n_epochs: int = 30,
    n_episodes: int = 1000,
) -> Policy:
    opt_state = optimiser.init(eqx.filter(policy, eqx.is_array))
    key = jax.random.PRNGKey(10)
    reward_log = tqdm(
        total=n_epochs,
        desc="Reward",
        position=2,
        leave=True,
        bar_format="{desc}",
    )
    rewards_to_show = []
    for epoch in tqdm(range(n_epochs), desc="Epochs", position=0, leave=True):
        epoch_rewards = 0
        for episode in tqdm(
            range(n_episodes), desc="Episodes", position=1, leave=False
        ):
            key, subkey = jax.random.split(key)
            dataset, info = gym_helpers.rollout_discrete(
                env, get_action, {"policy": policy}, subkey
            )
            dataloader = DataLoader(
                dataset, batch_size=4, shuffle=False, drop_last=True
            )

            epoch_rewards += jnp.sum(dataset.rewards.numpy())

            for batch in dataloader:
                b_states, b_actions, b_rewards, b_dones = batch
                b_states = jnp.array(b_states.numpy())
                b_actions = jnp.array(b_actions.numpy())
                b_rewards = jnp.array(b_rewards.numpy())
                b_dones = jnp.array(b_dones.numpy())

                policy, opt_state = step(
                    policy, b_states, b_actions, b_rewards, optimiser, opt_state
                )
        rewards_to_show.append(jnp.mean(epoch_rewards / n_episodes))
        reward_log.set_description_str(f"Last avg. rewards: {rewards_to_show[-1]}")
    plt.plot(rewards_to_show)
    plt.show()

    return policy
