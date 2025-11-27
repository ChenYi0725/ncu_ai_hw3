import os
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from training import train as train_pg
from testing import test as test_pg
from q_learning import QLearningAgent  
from taxi_env import TaxiEnv
from reward_wrapper import TaxiRewardWrapper
from sarsa_agent import SARSAAgent  


CONFIG_DIR = "compare_configs"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def print_config(config):
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("=" * 70)

def run_experiment(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)

    print(f"\nRunning config: {os.path.basename(config_path)}")
    print_config(config)
    
#   Policy Gradient

    pg_agent, _ = train_pg(
        n_episodes=config.get("n_episodes", 50000),
        max_steps=config.get("max_steps", 200),
        seed_start=config.get("seed_start", 0),
        seed_end=config.get("seed_end", 40000),
        verbose=config.get("verbose", True),
        reward_step=config.get("reward_step", -5),
        reward_delivery=config.get("reward_delivery", 20),
        reward_illegal=config.get("reward_illegal", -1),
        learning_rate=config.get("learning_rate", 0.01),
        value_lr=config.get("value_lr", 0.1),
        lr_decay=config.get("lr_decay", 0.99999),
        discount_factor=config.get("discount_factor", 0.99),
        entropy_coef=config.get("entropy_coef", 0.1),
    )
    
    pg_results = test_pg(
        model_filename=None,
        n_episodes=config.get("test_episodes", 100),
        seed_start=config.get("test_seed_start", 420000),
        verbose=True,
        reward_step=config.get("reward_step", -5),
        reward_delivery=config.get("reward_delivery", 20),
        reward_illegal=config.get("reward_illegal", -1),
        agent=pg_agent,
    )

    # Q-Learning

    env = TaxiRewardWrapper(TaxiEnv(),
                            reward_step=config.get("reward_step", -5),
                            reward_delivery=config.get("reward_delivery", 20),
                            reward_illegal=config.get("reward_illegal", -1))
    
    q_agent = QLearningAgent(
        n_states=env.observation_space.n,
        n_actions=env.action_space.n,
        learning_rate=config.get("learning_rate", 0.1),
        discount_factor=config.get("discount_factor", 0.99),
        epsilon=config.get("epsilon", 0.1),
        epsilon_decay=config.get("epsilon_decay", 0.999)
    )
    
    q_rewards = []
    for episode in range(config.get("n_episodes", 50000)):
        state, info = env.reset(seed=config.get("seed_start",0)+episode)
        total_reward = 0
        for _ in range(config.get("max_steps",200)):
            action = q_agent.choose_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            q_agent.update(state, action, reward, next_state, done)
            total_reward += reward
            state = next_state
            if done:
                break
        q_rewards.append(total_reward)
    env.close()
    

    env = TaxiRewardWrapper(TaxiEnv(),
                            reward_step=config.get("reward_step", -5),
                            reward_delivery=config.get("reward_delivery", 20),
                            reward_illegal=config.get("reward_illegal", -1))
    
    successes = 0
    steps_list = []
    for episode in range(config.get("test_episodes", 100)):
        state, info = env.reset(seed=config.get("test_seed_start",420000)+episode)
        step_count = 0
        for _ in range(config.get("max_steps",200)):
            action = np.argmax(q_agent.q_table[state])
            next_state, reward, terminated, truncated, info = env.step(action)
            step_count += 1
            if terminated or truncated:
                if reward == config.get("reward_delivery", 20):
                    successes += 1
                break
            state = next_state
        steps_list.append(step_count)
    env.close()
    
    q_avg_reward = np.mean(q_rewards)
    q_avg_steps = np.mean(steps_list)
    q_success_rate = successes / config.get("test_episodes",100)
    normalized_steps = q_avg_steps / config.get("max_steps",200)
    q_eval_score = q_success_rate*0.2 + (1-normalized_steps)*0.8
    
#========
    timestamp = datetime.now().strftime("%m%d_%H%M")
    config_name = os.path.splitext(os.path.basename(config_path))[0]
    
    pg_model_file = f"{MODEL_DIR}/{config_name}_PG_{timestamp}_{pg_results['evaluation_score']:.4f}.npy"
    q_model_file = f"{MODEL_DIR}/{config_name}_QL_{timestamp}_{q_eval_score:.4f}.npy"
    
    np.save(pg_model_file, pg_agent.theta)
    np.save(q_model_file, q_agent.q_table)
    
    print(f"PG model saved to {pg_model_file}")
    print(f"Q-Learning model saved to {q_model_file}")



def run_experiment(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)

    print(f"\nRunning config: {os.path.basename(config_path)}")
    print_config(config)
    
    # 1️⃣ Policy Gradient
    pg_agent, _ = train_pg(
        n_episodes=config.get("n_episodes", 50000),
        max_steps=config.get("max_steps", 200),
        seed_start=config.get("seed_start", 0),
        seed_end=config.get("seed_end", 40000),
        verbose=config.get("verbose", True),
        reward_step=config.get("reward_step", -5),
        reward_delivery=config.get("reward_delivery", 20),
        reward_illegal=config.get("reward_illegal", -1),
        learning_rate=config.get("learning_rate", 0.01),
        value_lr=config.get("value_lr", 0.1),
        lr_decay=config.get("lr_decay", 0.99999),
        discount_factor=config.get("discount_factor", 0.99),
        entropy_coef=config.get("entropy_coef", 0.1),
    )
    
    pg_results = test_pg(
        model_filename=None,
        n_episodes=config.get("test_episodes", 100),
        seed_start=config.get("test_seed_start", 420000),
        verbose=True,
        reward_step=config.get("reward_step", -5),
        reward_delivery=config.get("reward_delivery", 20),
        reward_illegal=config.get("reward_illegal", -1),
        agent=pg_agent,
    )

    #  Q-Learning
    env = TaxiRewardWrapper(TaxiEnv(),
                            reward_step=config.get("reward_step", -5),
                            reward_delivery=config.get("reward_delivery", 20),
                            reward_illegal=config.get("reward_illegal", -1))
    
    q_agent = QLearningAgent(
        n_states=env.observation_space.n,
        n_actions=env.action_space.n,
        learning_rate=config.get("learning_rate", 0.1),
        discount_factor=config.get("discount_factor", 0.99),
        epsilon=config.get("epsilon", 0.1),
        epsilon_decay=config.get("epsilon_decay", 0.999)
    )
    
    q_rewards = []
    for episode in range(config.get("n_episodes", 50000)):
        state, info = env.reset(seed=config.get("seed_start",0)+episode)
        total_reward = 0
        for _ in range(config.get("max_steps",200)):
            action = q_agent.choose_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            q_agent.update(state, action, reward, next_state, done)
            total_reward += reward
            state = next_state
            if done:
                break
        q_rewards.append(total_reward)
    env.close()
    
    env = TaxiRewardWrapper(TaxiEnv(),
                            reward_step=config.get("reward_step", -5),
                            reward_delivery=config.get("reward_delivery", 20),
                            reward_illegal=config.get("reward_illegal", -1))
    
    successes = 0
    steps_list = []
    for episode in range(config.get("test_episodes", 100)):
        state, info = env.reset(seed=config.get("test_seed_start",420000)+episode)
        step_count = 0
        for _ in range(config.get("max_steps",200)):
            action = np.argmax(q_agent.q_table[state])
            next_state, reward, terminated, truncated, info = env.step(action)
            step_count += 1
            if terminated or truncated:
                if reward == config.get("reward_delivery", 20):
                    successes += 1
                break
            state = next_state
        steps_list.append(step_count)
    env.close()
    
    q_avg_reward = np.mean(q_rewards)
    q_avg_steps = np.mean(steps_list)
    q_success_rate = successes / config.get("test_episodes",100)
    normalized_steps = q_avg_steps / config.get("max_steps",200)
    q_eval_score = q_success_rate*0.2 + (1-normalized_steps)*0.8

    #  SARSA
    env = TaxiRewardWrapper(TaxiEnv(),
                            reward_step=config.get("reward_step", -5),
                            reward_delivery=config.get("reward_delivery", 20),
                            reward_illegal=config.get("reward_illegal", -1))
    
    sarsa_agent = SARSAAgent(
        n_states=env.observation_space.n,
        n_actions=env.action_space.n,
        learning_rate=config.get("learning_rate", 0.1),
        discount_factor=config.get("discount_factor", 0.99),
        epsilon=config.get("epsilon", 0.1),
        epsilon_decay=config.get("epsilon_decay", 0.999)
    )
    
    sarsa_rewards = []
    for episode in range(config.get("n_episodes",50000)):
        state, info = env.reset(seed=config.get("seed_start",0)+episode)
        action = sarsa_agent.choose_action(state)
        total_reward = 0
        for _ in range(config.get("max_steps",200)):
            next_state, reward, terminated, truncated, info = env.step(action)
            next_action = sarsa_agent.choose_action(next_state)
            done = terminated or truncated
            sarsa_agent.update(state, action, reward, next_state, next_action, done)
            total_reward += reward
            state, action = next_state, next_action
            if done:
                break
        sarsa_rewards.append(total_reward)
    env.close()
    
    env = TaxiRewardWrapper(TaxiEnv(),
                            reward_step=config.get("reward_step", -5),
                            reward_delivery=config.get("reward_delivery", 20),
                            reward_illegal=config.get("reward_illegal", -1))
    
    successes = 0
    steps_list = []
    for episode in range(config.get("test_episodes",100)):
        state, info = env.reset(seed=config.get("test_seed_start",420000)+episode)
        step_count = 0
        for _ in range(config.get("max_steps",200)):
            action = np.argmax(sarsa_agent.q_table[state])
            next_state, reward, terminated, truncated, info = env.step(action)
            step_count += 1
            if terminated or truncated:
                if reward == config.get("reward_delivery", 20):
                    successes += 1
                break
            state = next_state
        steps_list.append(step_count)
    env.close()
    
    sarsa_avg_reward = np.mean(sarsa_rewards)
    sarsa_avg_steps = np.mean(steps_list)
    sarsa_success_rate = successes / config.get("test_episodes",100)
    normalized_steps = sarsa_avg_steps / config.get("max_steps",200)
    sarsa_eval_score = sarsa_success_rate*0.2 + (1-normalized_steps)*0.8

    #======== Save models
    timestamp = datetime.now().strftime("%m%d_%H%M")
    config_name = os.path.splitext(os.path.basename(config_path))[0]
    
    pg_model_file = f"{MODEL_DIR}/{config_name}_PG_{timestamp}_{pg_results['evaluation_score']:.4f}.npy"
    q_model_file = f"{MODEL_DIR}/{config_name}_QL_{timestamp}_{q_eval_score:.4f}.npy"
    sarsa_model_file = f"{MODEL_DIR}/{config_name}_SARSA_{timestamp}_{sarsa_eval_score:.4f}.npy"
    
    np.save(pg_model_file, pg_agent.theta)
    np.save(q_model_file, q_agent.q_table)
    np.save(sarsa_model_file, sarsa_agent.q_table)
    
    print(f"PG Eval Score: {pg_results['evaluation_score']:.4f}")
    print(f"Q-Learning Eval Score: {q_eval_score:.4f}")
    print(f"SARSA Eval Score: {sarsa_eval_score:.4f}")

    # Comparison plot
    plt.figure(figsize=(10,6))
    plt.bar(["PG","Q-Learning","SARSA"],
            [pg_results["evaluation_score"], q_eval_score, sarsa_eval_score])
    plt.ylabel("Evaluation Score")
    plt.title(f"{config_name} Comparison")
    plt.show()


def main_loop():
    config_files = [
        os.path.join(CONFIG_DIR, f)
        for f in os.listdir(CONFIG_DIR) if f.endswith(".json")
    ]
    if not config_files:
        print(f"No config files found in {CONFIG_DIR}")
        return
    for cfg in config_files:
        try:
            run_experiment(cfg)
        except Exception as e:
            print(f"Error running {cfg}: {e}")


if __name__ == "__main__":
    main_loop()
