import os
import json
import numpy as np
from datetime import datetime
from training import train
from testing import test

CONFIG_DIR = "configs"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)


def print_config(config):
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("=" * 70)


def run_from_config(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)

    print(f"\nRunning config: {os.path.basename(config_path)}")
    print_config(config)
    print("=" * 70)


    agent, _ = train(
        n_episodes=config.get("n_episodes", 50000),
        max_steps=200,
        seed_start=0,
        seed_end=40000,
        verbose=True,
        reward_step=config.get("reward_step", -5),
        reward_delivery=config.get("reward_delivery", 20),
        reward_illegal=config.get("reward_illegal", -1),
        learning_rate=config.get("learning_rate", 0.01),
        value_lr=config.get("value_lr", 0.1),
        lr_decay=config.get("lr_decay", 0.99999),
        discount_factor=config.get("discount_factor", 0.99),
        entropy_coef=config.get("entropy_coef", 0.1),
    )

 
    test_results = test(
        model_filename=None,  
        n_episodes=200,
        seed_start=42000,
        verbose=True,
        reward_step=config.get("reward_step", -5),
        reward_delivery=config.get("reward_delivery", 20),
        reward_illegal=config.get("reward_illegal", -1),
        agent=agent,  
    )

    evaluation_score = test_results["evaluation_score"]
    config_name = os.path.splitext(os.path.basename(config_path))[0]
    timestamp = datetime.now().strftime("%m%d_%H%M")
    model_filename = f"{MODEL_DIR}/{config_name}_{timestamp}_{evaluation_score:.4f}.npy"
    np.save(model_filename, agent.theta)
    print(f"model saved to {model_filename}")
    print(f"evaluation score: {evaluation_score:.4f}")


def main_loop():
    config_files = [
        os.path.join(CONFIG_DIR, f)
        for f in os.listdir(CONFIG_DIR)
        if f.endswith(".json")
    ]

    if not config_files:
        print(f"No config files found in {CONFIG_DIR}")
        return


    for cfg in config_files:
        try:
            run_from_config(cfg)
        except Exception as e:
            print(f"Error running {cfg}: {e}")


if __name__ == "__main__":
    main_loop()
