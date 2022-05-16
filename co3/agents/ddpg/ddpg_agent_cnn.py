import itertools as it
import logging
import math
import os
from collections import deque, namedtuple
from contextlib import contextmanager
from random import choices

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from agent.continuous_action_space_agent import ContinuousActionSpaceAgent
from constants import DEVICE
from continuous_action_space_model import ContinuousActionSpaceModel
from networks.fully_connected import FullyConnected
from project_functions import initialize_weights
from pydantic_config import DdpgProcessConfig
from replay_buffer import ReplayBuffer
from sentient_util.exceptions import InvalidConfiguration
from torch import Tensor

_log = logging.critical

torch.set_printoptions(precision=10, sci_mode=True)


class Actor(torch.nn.Module):
    def __init__(self, env, config: DdpgProcessConfig):
        super().__init__()

        def cnn_shape(layer: nn.Module):
            with torch.no_grad():
                return layer(torch.randn((1, *shape))).squeeze(0).shape

        shape = env.env.observation_space.shape

        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=shape[0],
                out_channels=32,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=1, stride=1),
            # torch.nn.Dropout(p=1 - 0),
        )

        shape = cnn_shape(self.layer1)

        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=shape[0],
                out_channels=shape[1],
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=1, stride=1),
            # torch.nn.Dropout(p=1 - 0),
        )

        shape = cnn_shape(self.layer2)

        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=shape[0],
                out_channels=64,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=1, stride=1, padding=0),
            # torch.nn.Dropout(p=1 - 0),
        )

        shape = cnn_shape(self.layer3)

        self.fc1 = torch.nn.Linear(math.prod(shape), 64, bias=True)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.layer4 = torch.nn.Sequential(
            self.fc1, torch.nn.ReLU(), torch.nn.Dropout(p=1 - 0)
        )

        self.fc2 = torch.nn.Linear(64, 2, bias=True)
        torch.nn.init.xavier_uniform_(self.fc2.weight)  # initialize parameters

    def forward(self, x):

        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)  # Flatten for the FC layers
        out = self.fc1(out)
        out = self.fc2(out)

        return out


class Critic(ContinuousActionSpaceModel):
    def __init__(self, env, config: DdpgProcessConfig):

        super().__init__(env)

        if (model_config := getattr(config.agent.torch_models, "Critic")) is None:
            raise InvalidConfiguration("Critic Not Defined")

        self._model = FullyConnected(
            input_size=self.state_size,
            hidden_dims=model_config.hidden_dims,
            activations=model_config.activations,
        )

        self.action_value = nn.Linear(
            env.action_space.shape[0], model_config.hidden_dims[-1]
        )
        self.output_layer = nn.Linear(model_config.hidden_dims[-1], 1)

        self.initialize_weights()

        self.to(DEVICE)

    def forward(self, state: Tensor, action: Tensor) -> Tensor:

        state_value = self._model(state)

        action_value = F.relu(self.action_value(action))

        state_action_value = F.relu(torch.add(state_value, action_value))

        return self.output_layer(state_action_value)


class Agent(ContinuousActionSpaceAgent):
    """"""

    def __init__(self, *, config: DdpgProcessConfig, env: gym):

        super().__init__(config=config, env=env, actor_class=Actor, critic_class=Critic)

        self.replay_buffer = ReplayBuffer(
            batch_size=config.agent.batch_size, buffer_size=config.agent.buffer_size
        )

    def select_action(self, state: Tensor) -> Tensor:

        with torch.no_grad():

            action = self.actor(state.unsqueeze(0)) + self.noise()

            # return torch.clamp(
            #     action.squeeze(0), self.action_space_low, self.action_space_high
            # )

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
