#!/usr/bin/env python

# https://github.com/senya-ashukha/quantile-regression-dqn-pytorch

import importlib
import logging
import os
import random

import numpy as np
import torch
import torch.nn as nn
from agents.dqn.dqn_base import DqnBase
from decorators import call_counter
from network_trace import NetworkTrace
from replay_buffer import ReplayBuffer
from rewards_csv import EvaluateRewardsCsv
from torch import tensor

np.set_printoptions(precision=6)  # type:ignore


try:
    profile  # type:ignore
except NameError:
    profile = lambda x: x


class AbstractAgent(DqnBase):
    """"""

    def __init__(self, *, config, env):

        try:

            network = importlib.import_module(
                "dqn.qrdqn_network"
            ).QrdqnNetwork  # type:ignore

            super().__init__(network, config=config, env=env)

            self.replay_buffer = ReplayBuffer(
                batch_size=config.agent.batch_size, buffer_size=config.agent.buffer_size
            )

            self.tau = tensor(
                (2 * np.arange(self.actor.quantile_resolution) + 1)
                / (2.0 * self.actor.quantile_resolution), # type: ignore
            ).view(1, -1)

            logging.debug(f"QrdqnAgent Initialized")

        except BaseException as msg:
            raise

    def __enter__(self):
        self.rewards_csv = EvaluateRewardsCsv(config=self.config)
        return self

    def __exit__(self, type, value, traceback):
        self.rewards_csv.close()
        self.network_trace.close()
        if type or value or traceback:
            logging.error(f"__exit__ traceback: {type=}, {value=}, {traceback=}")

    def select_action(self, *, state):

        return self.actor.select_action(
            state=state, training=self.config.agent.training, eps=self.eps # type: ignore
        )

    def update_target(self):
        self.actor_target.load_state_dict(self.actor.state_dict())

    def train(self, steps_done):

        config = self.config

        batch_size = self.config.agent.batch_size # type: ignore

        states, actions, next_states, rewards, dones = self.replay_buffer.sample()

        theta = self.actor(states)[np.arange(batch_size), [a for a in actions.long()]]

        Znext = self.actor_target(next_states).detach()

        Znext_max = Znext[np.arange(batch_size), Znext.mean(2).max(1)[1]]

        Ttheta = rewards + config.agent.gamma * (1 - dones) * Znext_max

        diff = Ttheta.t().unsqueeze(-1) - theta

        huber = lambda x, k=1.0: torch.where(
            x.abs() < k, 0.5 * x.pow(2), k * (x.abs() - 0.5 * k)
        )

        loss = huber(diff) * (self.tau - (diff.detach() < 0).float()).abs()
        loss = loss.mean()

        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

        if steps_done % config.agent.target_update_interval == 0:
            self.update_target()

