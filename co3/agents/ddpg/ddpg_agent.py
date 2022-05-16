from dataclasses import dataclass
import logging
from typing import Any

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from agent.continuous_action_space_agent import ContinuousActionSpaceAgent
from constants import DEVICE
from continuous_action_space_model import ContinuousActionSpaceModel
from networks.fully_connected import FullyConnected
from pydantic_config import DdpgProcessConfig
from replay_buffer import ReplayBuffer
from sentient_util.exceptions import InvalidConfiguration
from torch import Tensor

_log = logging.critical

torch.set_printoptions(precision=10, sci_mode=True)


@dataclass(eq=False, order=False)
class Actor(ContinuousActionSpaceModel):
    env: Any
    config: DdpgProcessConfig

    def __post_init__(self):

        super().__init__(self.env)

        torch_model = self.config.agent.torch_models.Actor

        self._model = FullyConnected(
            input_size=self.state_size,
            hidden_dims=list(torch_model.hidden_dims),
            output_size=self.action_size,
            activations=list(torch_model.activations),
        )

        self.initialize_weights()

        self.to(DEVICE)

        self.train() if self.config.agent.training else self.eval()

    def forward(self, state: Tensor) -> Tensor:
        return torch.tanh(self._model(state)) * self.action_scale + self.action_bias


@dataclass(eq=False, order=False)
class Critic(ContinuousActionSpaceModel):
    env: Any
    config: DdpgProcessConfig

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

    def __init__(self, *, config: DdpgProcessConfig, env: gym.Env):

        logging.critical(env)

        super().__init__(config=config, env=env, actor_class=Actor, critic_class=Critic)

        self.replay_buffer = ReplayBuffer(
            batch_size=config.agent.batch_size,
            buffer_size=config.agent.buffer_size,
        )

    def predicted(self, state: Tensor, action: Tensor):
        return self.critic(state, action).item()

    def train(self, *args):

        states, actions, next_states, rewards, dones = self.replay_buffer.sample()

        # Flatten state and next_states to vectors
        states = states.flatten(start_dim=1)
        next_states = next_states.flatten(start_dim=1)

        with torch.no_grad():

            targets: Tensor = rewards + self.gamma * (1 - dones) * self.critic_target(
                next_states, self.actor_target(next_states)
            )

        self.optimize_critic(
            critic_loss := self.loss(targets, self.critic(states, actions))
        )
        logging.debug(f"{critic_loss.item()=}")

        self.optimize_actor(
            actor_loss := -self.critic(states, self.actor(states)).mean()
        )
        logging.debug(f"{actor_loss.item()=}")

        if next(self.training_counter) % self.target_update_interval == 0:
            super().soft_update(self.critic_target, self.critic)
            super().soft_update(self.actor_target, self.actor)

    def __str__(self):
        return str(self.actor) + str(self.critic)
