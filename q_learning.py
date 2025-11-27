import numpy as np


class QLearningAgent:
    def __init__(
        self,
        n_states,
        n_actions,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=0.1,
        epsilon_decay=0.999,
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.q_table = np.zeros((n_states, n_actions))

    def choose_action(self, state, action_mask=None):
        """Epsilon-greedy action selection with action mask."""
        if action_mask is None:
            action_mask = np.ones(self.n_actions, dtype=bool)

        available_actions = np.where(action_mask)[0]

        if np.random.rand() < self.epsilon:
            return int(np.random.choice(available_actions))
        else:
            q_values = self.q_table[state].copy()
            q_values[~action_mask] = -np.inf
            return int(np.argmax(q_values))

    def update(self, state, action, reward, next_state, done, next_action_mask=None):
        """Q-Learning update with action mask support."""
        best_next_action = (
            np.argmax(np.where(next_action_mask, self.q_table[next_state], -np.inf))
            if next_action_mask is not None
            else np.argmax(self.q_table[next_state])
        )

        target = reward + (
            0 if done else self.gamma * self.q_table[next_state, best_next_action]
        )
        self.q_table[state, action] += self.lr * (target - self.q_table[state, action])
        self.epsilon *= self.epsilon_decay
