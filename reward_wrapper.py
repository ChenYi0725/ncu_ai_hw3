import gym
import numpy as np


class TaxiRewardWrapper(gym.Wrapper):
    """
    Reward wrapper for Taxi environment
    Allows customizing reward values without modifying the original environment
    """
    def __init__(self, env, reward_step=-5, reward_delivery=20, reward_illegal=-1):
        super().__init__(env)
        self.custom_reward_step = reward_step
        self.custom_reward_delivery = reward_delivery
        self.custom_reward_illegal = reward_illegal

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        if reward == self.env.reward_delivery:
            reward = self.custom_reward_delivery
        elif reward == self.env.reward_illegal:
            reward = self.custom_reward_illegal
        elif reward == self.env.reward_step:
            reward = self.custom_reward_step

        return obs, reward, terminated, truncated, info