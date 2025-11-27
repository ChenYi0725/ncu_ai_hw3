import numpy as np
from taxi_env import TaxiEnv
from reward_wrapper import TaxiRewardWrapper
from policy_gradient_agent import PolicyGradientAgentOptimized


def train(
    n_episodes=50000,
    max_steps=200,
    seed_start=0,
    seed_end=40000,
    verbose=True,
    reward_step=-5,
    reward_delivery=20,
    reward_illegal=-1,
    learning_rate=0.01,
    value_lr=0.1,
    lr_decay=0.99999,
    discount_factor=0.99,
    entropy_coef=0.1,
):

    env = TaxiRewardWrapper(
        TaxiEnv(),
        reward_step=reward_step,
        reward_delivery=reward_delivery,
        reward_illegal=reward_illegal,
    )
    print("learning_rate:", learning_rate)
    agent = PolicyGradientAgentOptimized(
        n_states=env.observation_space.n,
        n_actions=env.action_space.n,
        learning_rate=learning_rate,
        value_lr=value_lr,
        lr_decay=lr_decay,
        discount_factor=discount_factor,
        entropy_coef=entropy_coef,
    )

    episode_rewards = []
    success_count = 0

    if verbose:
        print(f"training episodes: {n_episodes}")
        print("=" * 70)

    for episode in range(n_episodes):
        state, info = env.reset(seed=seed_start + episode % seed_end)
        total_reward = 0
        episode_history = []

        for step in range(max_steps):
       
            action_mask = env.action_mask(state)
            action = agent.choose_action(state, action_mask=action_mask)

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_history.append((state, action, reward))
            total_reward += reward
            state = next_state

            if done:
                if terminated and reward == reward_delivery:
                    success_count += 1
                break

        agent.update(episode_history)
        episode_rewards.append(total_reward)

        if verbose and (episode + 1) % 1000 == 0:
            avg_reward = np.mean(episode_rewards[-1000:])
            success_rate = success_count / 1000
            print(
                f"episode {episode+1}/{n_episodes} | avg reward: {avg_reward:.2f} | success rate: {success_rate:.1%} | lr: {agent.lr:.6f}"
            )
            success_count = 0

    if verbose:
        print("=" * 70)
        print("Training completed!")

    env.close()
    return agent, episode_rewards
