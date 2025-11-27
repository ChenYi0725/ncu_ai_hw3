import numpy as np
from taxi_env import TaxiEnv
from reward_wrapper import TaxiRewardWrapper
from policy_gradient_agent import PolicyGradientAgentOptimized


def test(
    model_filename,
    n_episodes=100,
    seed_start=42,
    verbose=True,
    reward_step=-5,
    reward_delivery=20,
    reward_illegal=-1,
    agent=None,
):

    env = TaxiRewardWrapper(
        TaxiEnv(),
        reward_step=reward_step,
        reward_delivery=reward_delivery,
        reward_illegal=reward_illegal,
    )

    if agent is None:
        if model_filename is None:
            raise ValueError("Either model_filename or agent must be provided")
        agent = PolicyGradientAgentOptimized(
            n_states=env.observation_space.n, n_actions=env.action_space.n
        )
        agent.theta = np.load(model_filename)

    rewards = []
    steps_list = []
    successes = 0

    for episode in range(n_episodes):
        state, info = env.reset(seed=seed_start + episode)
        episode_reward = 0
        step_count = 0

        for step in range(200):
        
            action_mask = info.get("action_mask", np.ones(env.action_space.n))
            masked_theta = np.where(action_mask, agent.theta[state], -np.inf)
            action = int(np.argmax(masked_theta))

            next_state, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward
            step_count += 1

            if terminated or truncated:
                if reward == reward_delivery:
                    successes += 1
                break

            state = next_state

        rewards.append(episode_reward)
        steps_list.append(step_count)

    avg_reward = np.mean(rewards)
    avg_steps = np.mean(steps_list)
    success_rate = successes / n_episodes


    normalized_steps = avg_steps / 200
    step_score = 1 - normalized_steps
    evaluation_score = success_rate * 0.2 + step_score * 0.8

    if verbose:
        print(f"\ntest result (seed {seed_start}-{seed_start+n_episodes-1}):")
        print("=" * 70)
        print(f"   avg reward: {avg_reward:.2f}")
        print(f"   avg steps: {avg_steps:.2f}")
        print(f"   success rate: {success_rate:.1%}")
        print(
            f"   evaluation score: {evaluation_score:.4f} ({evaluation_score*100:.2f}%)"
        )
        print("=" * 70)

    env.close()

    return {
        "avg_reward": avg_reward,
        "avg_steps": avg_steps,
        "success_rate": success_rate,
        "evaluation_score": evaluation_score,
    }
