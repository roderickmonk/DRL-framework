from typing import Callable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from constants import DEVICE
from continuous_action_space_model import ContinuousActionSpaceModel
from networks.fully_connected import FullyConnected
from project_functions import initialize_weights
from torch import Tensor
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6


class Actor(ContinuousActionSpaceModel):
    def __init__(self, env, config):

        super().__init__(env)

        model_config = config.agent.torch_models.Actor

        self.actor = FullyConnected(
            input_size=self.state_size,
            hidden_dims=(hidden_dims := model_config.hidden_dims),
            output_size=model_config.hidden_dims[-1],
            activations=model_config.activations,
        )

        self.mean_linear = nn.Linear(hidden_dims[-1], self.action_size)
        self.log_std_linear = nn.Linear(hidden_dims[-1], self.action_size)

        initialize_weights(self)

        self.train() if config.agent.training else self.eval()

        self.to(DEVICE)

    def forward(self, state) -> Tuple[Tensor, Tensor]:

        x = self.actor(state)

        mean: Tensor = self.mean_linear(x)
        log_std = self.log_std_linear(x)

        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)

        return mean, log_std

    def sample(self, state: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)

        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


class Critic(ContinuousActionSpaceModel):
    def __init__(self, env, config):

        super().__init__(env)

        model_config = config.agent.torch_models.Critic

        self.Q1 = FullyConnected(
            input_size=self.state_size + self.action_size,
            hidden_dims=model_config.hidden_dims,
            output_size=self.action_size,
            activations=model_config.activations,
        )

        self.Q2 = FullyConnected(
            input_size=self.state_size + self.action_size,
            hidden_dims=model_config.hidden_dims,
            output_size=self.action_size,
            activations=model_config.activations,
        )

        initialize_weights(self)

        self.to(DEVICE)

    def forward(self, state, action) -> Tuple[Tensor, Tensor]:
        xu = torch.cat([state, action], 1)
        return self.Q1(xu), self.Q2(xu)
