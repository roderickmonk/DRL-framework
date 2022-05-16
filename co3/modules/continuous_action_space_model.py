import math

import torch
from torch import nn

from constants import DEVICE
from project_functions import float_tensor


class ContinuousActionSpaceModel(nn.Module):
    def __init__(self, env):

        super().__init__()

        self.state_size = math.prod(env.observation_space.shape)
        self.action_size = env.action_space.shape[0]

        self.action_space_low = env.action_space.low[0]
        self.action_space_high = env.action_space.high[0]

        self.action_scale = float_tensor(
            (env.action_space.high - env.action_space.low) / 2.0
        )
        self.action_bias = float_tensor(
            (env.action_space.high + env.action_space.low) / 2.0
        )

    def initialize_weights(self):

        if isinstance(self, nn.Conv1d):
            nn.init.kaiming_uniform_(self.weight.data, nonlinearity="relu")
            if self.bias is not None:
                nn.init.constant_(self.bias.data, 0)
        elif isinstance(self, nn.Linear):
            nn.init.kaiming_uniform_(self.weight.data)
            nn.init.constant_(self.bias.data, 0)

