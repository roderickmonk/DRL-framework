import gym
import numpy as np


class ActionWrapper(gym.ActionWrapper):
    """"""

    def __init__(self, env):

        super().__init__(env)

        self.low = env.action_space.low
        self.high = env.action_space.high

        self.scale = (self.high - self.low) / 2.0
        self.bias = (self.high + self.low) / 2.0

    def action(self, action):

        action = action * self.scale + self.bias
        action = np.clip(action, self.low, self.high)
        # print (f"out {action=}")
        return action

