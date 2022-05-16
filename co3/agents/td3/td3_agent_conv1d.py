import copy
import itertools as it
import logging
import os
from typing import Any, Tuple

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from agent.continuous_action_space_agent import ContinuousActionSpaceAgent
from constants import DEVICE
from continuous_action_space_model import ContinuousActionSpaceModel
from networks.fully_connected import FullyConnected
from omegaconf import OmegaConf
from replay_buffer import ReplayBuffer
from sentient_util.exceptions import InvalidConfiguration
from torch import Tensor
from pydantic_config import Td3ProcessConfig
from project_functions import float_tensor
from networks.actor_conv1d import Actor

torch.set_printoptions(precision=10, sci_mode=True)


class Critic(ContinuousActionSpaceModel):
    def __init__(self, env, config: Td3ProcessConfig):

        super().__init__(env)

        if (model_config := getattr(config.agent.torch_models, "Critic")) is None:
            raise InvalidConfiguration("Critic Not Defined")

        self._Q1 = FullyConnected(
            input_size=self.state_size + self.action_size,
            hidden_dims=model_config.hidden_dims,
            output_size=1,
            activations=model_config.activations,
        )

        self._Q2 = FullyConnected(
            input_size=self.state_size + self.action_size,
            hidden_dims=model_config.hidden_dims,
            output_size=1,
            activations=model_config.activations,
        )

        self.initialize_weights()

        self.to(DEVICE)

    def forward(self, state: Tensor, action: Tensor) -> Tuple[Tensor, Tensor]:

        # sa = torch.cat([state, action], 1)
        return self._Q1(sa := torch.cat([state, action], 1)), self._Q2(sa)

    def Q1(self, state: Tensor, action: Tensor) -> Tensor:
        return self._Q1(torch.cat([state, action], 1))


class Agent(ContinuousActionSpaceAgent):
    def __init__(self, *, config: Td3ProcessConfig, env):

        super().__init__(
            config=config,
            env=env,
            actor_class=Actor,
            critic_class=Critic,
        )

        self.config = config

        self.replay_buffer = ReplayBuffer(
            batch_size=config.agent.batch_size,
            buffer_size=config.agent.buffer_size,
        )

        # Agent-specific parameters
        self.policy_noise = float_tensor(config.agent.policy_noise)

        self.noise_clip: float = config.agent.noise_clip

    def predicted(self, state, action):
        return self.critic(
            float_tensor(state).unsqueeze(0),
            float_tensor(action).unsqueeze(0),
        )[0].item()

    def select_action(self, state: Tensor) -> Tensor:

        with torch.no_grad():

            action = self.actor(state.unsqueeze(0)) + self.noise()
            return action.squeeze(0)

    def train(self, *args):

        states, actions, next_states, rewards, dones = self.replay_buffer.sample()

        # Select action according to policy and add clipped noise
        with torch.no_grad():

            noise = (
                torch.randn_like(actions) * self.policy_noise * self.action_scale
            ).clamp(-self.noise_clip, self.noise_clip)

            next_actions = self.actor_target(next_states).clamp(
                self.action_space_low, self.action_space_high
            )

            next_actions = (next_actions + noise).clamp(
                self.action_space_low, self.action_space_high
            )

            target_Q = rewards + (1 - dones) * self.gamma * torch.min(
                *self.critic_target(next_states.flatten(start_dim=1), next_actions)
            )

        current_Q1, current_Q2 = self.critic(states.flatten(start_dim=1), actions)

        self.optimize_critic(
            critic_loss := self.loss(current_Q1, target_Q)
            + self.loss(current_Q2, target_Q)
        )
        # logging.debug(f"{critic_loss.item()=}")

        self.optimize_actor(
            actor_loss := -self.critic.Q1(
                states.flatten(start_dim=1), self.actor(states)
            ).mean()  # type:ignore
        )
        # logging.debug(f"{actor_loss.item()=}")

        if next(self.training_counter) % self.target_update_interval == 0:

            super().soft_update(self.critic_target, self.critic)
            super().soft_update(self.actor_target, self.actor)
