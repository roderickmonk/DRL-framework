import logging
import os
from copy import deepcopy
from modulefinder import Module

import torch
import torch.nn as nn
import torch.nn.functional as F
from agent.continuous_action_space_agent import ContinuousActionSpaceAgent
from constants import DEVICE
from replay_buffer import ReplayBuffer
from torch import Tensor, tensor
from torch.optim import Adam

_log = logging.debug

# https://github.com/pranz24/pytorch-soft-actor-critic

from typing import Callable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from constants import DEVICE
from continuous_action_space_model import ContinuousActionSpaceModel
from networks.actor_conv1d import Actor as Conv1dActor
from networks.fully_connected import FullyConnected
from project_functions import initialize_weights
from pydantic_config import TD3_SimEnv1_Config
from torch import Tensor
from torch.distributions import Normal
from devtools import debug

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6


class Actor(ContinuousActionSpaceModel):
    def __init__(self, env, config: TD3_SimEnv1_Config):

        super().__init__(env)

        # Modifying config, so need to make a copy
        config = deepcopy(config)

        model_config = config.agent.torch_models.Actor

        debug(model_config)

        if len(model_config.hidden_dims) < 2:
            raise RuntimeError(
                f"SAC requires at least 2 hidden dims, found {len(model_config.hidden_dims)}"
            )

        last_hidden_dim = model_config.hidden_dims[-1]

        self.actor = Conv1dActor(env, config, output_size_explicit=False)

        self.mean_linear = nn.Linear(last_hidden_dim, self.action_size)
        self.log_std_linear = nn.Linear(last_hidden_dim, self.action_size)

        self.initialize_weights()

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

        self.initialize_weights()

        self.to(DEVICE)

    def forward(self, state, action) -> Tuple[Tensor, Tensor]:
        xu = torch.cat([state, action], 1)
        return self.Q1(xu), self.Q2(xu)


class Agent(ContinuousActionSpaceAgent):
    def __init__(self, *, config, env):

        super().__init__(config=config, env=env, actor_class=Actor, critic_class=Critic)

        self.replay_buffer = ReplayBuffer(
            batch_size=config.agent.batch_size,
            buffer_size=config.agent.buffer_size,
        )

        # Agent-specific params
        self.alpha = config.agent.alpha
        self.automatic_entropy_tuning = config.agent.automatic_entropy_tuning

        # Target Entropy = −dim(A) (e.g. -6 for HalfCheetah-v2) as given in the paper
        if self.automatic_entropy_tuning is True:
            self.target_entropy = -torch.prod(
                tensor(self.env.action_space.shape)  # type:ignore
            ).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=DEVICE)
            self.alpha_optimizer = Adam([self.log_alpha], lr=config.agent.actor_lr)

    def select_action(self, state: Tensor) -> Tensor:

        breakpoint()

        # state = state.reshape(1, -1)

        if self.actor.training is False:
            action, _, _ = self.actor.sample(state.unsqueeze(0))  # type:ignore
        else:
            _, _, action = self.actor.sample(state.unsqueeze(0))  # type:ignore

        return action.squeeze(0)

    def predicted(self, state, action):
        return self.critic(
            tensor(state.flatten(start_dim=1)).unsqueeze(0),
            tensor(action).unsqueeze(0),
        )[0].item()

    def train(self, *args):

        states, actions, next_states, rewards, dones = self.replay_buffer.sample()

        with torch.no_grad():

            next_state_action, next_state_log_pi, _ = self.actor.sample(
                next_states
            )  # type:ignore
            qf1_next_target, qf2_next_target = self.critic_target(
                next_states.flatten(start_dim=1), next_state_action
            )
            min_qf_next_target = (
                torch.min(qf1_next_target, qf2_next_target)
                - self.alpha * next_state_log_pi
            )
            target_Q = rewards + (1 - dones) * self.gamma * (min_qf_next_target)

        self.critic_optimizer.zero_grad()

        # Two Q-functions to mitigate positive bias in the actor improvement step
        qf1, qf2 = self.critic(states.flatten(start_dim=1), actions)

        # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf1_loss = self.loss(qf1, target_Q)

        # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = self.loss(qf2, target_Q)

        qf_loss = qf1_loss + qf2_loss

        # ToDo: Why is the following needed?
        try:
            qf_loss.backward()
        except RuntimeError as msg:
            # logging.error (msg)
            pass

        self.critic_optimizer.step()

        pi, log_pi, _ = self.actor.sample(states)  # type:ignore

        qf1_pi, qf2_pi = self.critic(states.flatten(start_dim=1), pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
        actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(
                self.log_alpha * (log_pi + self.target_entropy).detach()
            ).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()

        else:
            alpha_loss = tensor([0.0])

        if next(self.training_counter) % self.target_update_interval == 0:
            super().soft_update(self.critic_target, self.critic)
