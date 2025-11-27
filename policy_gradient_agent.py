import numpy as np
import sys
from pathlib import Path


class PolicyGradientAgentOptimized:

    def __init__(self, n_states, n_actions,
                 learning_rate=0.01,
                 value_lr=0.1,
                 lr_decay=0.9999,
                 lr_min=0.0001,
                 discount_factor=0.99,
                 entropy_coef=0.01):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = learning_rate
        self.lr_min = lr_min
        self.lr_decay = lr_decay
        self.value_lr = value_lr
        self.gamma = discount_factor
        self.entropy_coef = entropy_coef
        self.theta = np.zeros((n_states, n_actions))
        self.V = np.zeros(n_states)
        self.episode_count = 0

    def get_policy(self, state):
        theta_state = self.theta[state] - np.max(self.theta[state])
        exp_theta = np.exp(theta_state)
        return exp_theta / np.sum(exp_theta)

    def choose_action(self, state, action_mask=None):
        policy = self.get_policy(state)
        if action_mask is not None:
            policy = policy * action_mask
            if policy.sum() == 0:
                policy = np.ones_like(policy)  
            policy /= policy.sum()
        return np.random.choice(self.n_actions, p=policy)

    def update(self, episode_history):
        if len(episode_history) == 0:
            return

        returns = []
        G = 0
        for state, action, reward in reversed(episode_history):
            G = reward + self.gamma * G
            returns.insert(0, G)
        returns = np.array(returns)

        for t, (state, action, reward) in enumerate(episode_history):
            td_error = returns[t] - self.V[state]
            self.V[state] += self.value_lr * td_error

        advantages = np.array([returns[t] - self.V[state] for t, (state, action, reward) in enumerate(episode_history)])
        if len(advantages) > 1:
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-9)

        for t, (state, action, reward) in enumerate(episode_history):
            policy = self.get_policy(state)
            grad = np.zeros(self.n_actions)
            grad[action] = 1.0
            grad -= policy
            entropy_grad = -np.log(policy + 1e-9) - 1
            total_grad = advantages[t] * grad + self.entropy_coef * entropy_grad
            self.theta[state] += self.lr * total_grad

        self.lr = max(self.lr_min, self.lr * self.lr_decay)
        self.episode_count += 1