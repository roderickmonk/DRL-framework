import itertools as it
import logging
import math
import os
from collections import deque, namedtuple
from contextlib import contextmanager
from dataclasses import dataclass, field
from random import choices
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from agent.continuous_action_space_agent import ContinuousActionSpaceAgent
from constants import DEVICE
from continuous_action_space_model import ContinuousActionSpaceModel
from networks.fully_connected import FullyConnected
from networks.actor_conv1d import Actor
from project_functions import initialize_weights
from pydantic_config import DDPG_SimEnv1_Config, TD3_SimEnv1_Config
from pydantic_config import Network
from replay_buffer import ReplayBuffer
from torch import Tensor, tensor

_log = logging.critical

torch.set_printoptions(precision=10, sci_mode=True)

ProcessConfigType = DDPG_SimEnv1_Config | TD3_SimEnv1_Config


@dataclass(eq=False, order=False)
class Critic(ContinuousActionSpaceModel):
    env: Any
    config: ProcessConfigType

    def __post_init__(self):

        super().__init__(self.env)

        torch_model = self.config.agent.torch_models.Critic

        self._model = FullyConnected(
            input_size=self.state_size,
            hidden_dims=list(torch_model.hidden_dims),
            activations=list(torch_model.activations),
        )

        self.action_value = nn.Linear(
            self.env.action_space.shape[0], torch_model.hidden_dims[-1]
        )
        self.output_layer = nn.Linear(torch_model.hidden_dims[-1], 1)

        self.initialize_weights()

        self.to(DEVICE)

    def forward(self, state: Tensor, action: Tensor) -> Tensor:

        state_value = self._model(state)

        action_value = F.relu(self.action_value(action))

        state_action_value = F.relu(torch.add(state_value, action_value))

        return self.output_layer(state_action_value)


class Agent(ContinuousActionSpaceAgent):
    """"""

    def __init__(self, *, config: ProcessConfigType, env):

        super().__init__(config=config, env=env, actor_class=Actor, critic_class=Critic)

        self.replay_buffer = ReplayBuffer(
            batch_size=config.agent.batch_size, buffer_size=config.agent.buffer_size
        )

    def select_action(self, state: Tensor) -> Tensor:

        with torch.no_grad():

            action = self.actor(state.unsqueeze(0)) + self.noise()
            return action.squeeze(0)

    def predicted(self, state: Tensor, action: Tensor):
        return self.critic(state, action).item()

    def train(self, *args):

        with torch.no_grad():

            states, actions, next_states, rewards, dones = self.replay_buffer.sample()

            targets: Tensor = rewards + self.gamma * (1 - dones) * self.critic_target(
                next_states.flatten(start_dim=1), self.actor_target(next_states)
            )

        self.optimize_critic(
            critic_loss := self.loss(
                targets, self.critic(states.flatten(start_dim=1), actions)
            )
        )
        logging.debug(f"{critic_loss.item()=}")

        self.optimize_actor(
            actor_loss := -self.critic(
                states.flatten(start_dim=1), self.actor(states)
            ).mean()
        )
        logging.debug(f"{actor_loss.item()=}")

        if next(self.training_counter) % self.target_update_interval == 0:
            super().soft_update(self.critic_target, self.critic)
            super().soft_update(self.actor_target, self.actor)

    def __str__(self):
        return str(self.actor) + str(self.critic)
