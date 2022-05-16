# https://github.com/pytorch/tutorials
import importlib
import logging
import os
import random

import gym

# import tracemalloc
import numpy as np
import torch
import util
from agent.agent import AbstractAgent
from constants import DEVICE
from replay_buffer import ReplayBuffer
from rewards_csv import EvaluateRewardsCsv
from torch.nn import functional as F

from agents.dqn.dqn_base import DqnBase

torch.set_printoptions(edgeitems=100, linewidth=1000)


class Agent(DqnBase):
    """"""

    def __init__(self, *, config, env):
        """"""

        network = importlib.import_module("dqn.dqn_network").DqnNetwork  # type:ignore

        super().__init__(network, config=config, env=env)

        if config.agent.training:
            self.actor.train()
        else:
            self.actor.eval()

        self.replay_buffer = ReplayBuffer(
            batch_size=config.agent.batch_size, buffer_size=config.agent.buffer_size
        )

        logging.debug(f"DqnAgent Initialized")

    def select_action(self, *, state):

        self.actor.eval()
        with torch.no_grad():
            action_values = self.actor(state)
        self.actor.train()

        # Epsilon-greedy action selection
        if self.eps and (random.random() > self.eps or not self.config.agent.training):
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.env.action_space.n))

    def train(self, steps_done):

        config = self.config

        states, actions, next_states, rewards, dones = self.replay_buffer.sample()

        # Get max predicted Q values (for next states) from target model

        Q_targets_next = self.actor_target(next_states).detach().max(1)[0].unsqueeze(1)

        # Compute Q targets for current states
        Q_targets = rewards + config.agent.gamma * (1 - dones) * Q_targets_next

        # Get expected Q values from local model
        Q_expected = self.actor(states).gather(1, actions.long())

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

        if steps_done % config.agent.target_update_interval == 0:
            super().soft_update(self.actor_target, self.actor)

    def __enter__(self):
        self.rewards_csv = EvaluateRewardsCsv(config=self.config)
        return self

    def __exit__(self, type, value, traceback):
        self.rewards_csv.close()
        if type or value or traceback:
            logging.error(f"__exit__ traceback: {type=}, {value=}, {traceback=}")

