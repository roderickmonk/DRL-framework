# Derived from: https://github.com/pytorch/examples/blob/master/reinforcement_learning/actor_critic.py

import logging
import os
import random
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch
from agent.agent import AbstractAgent
from sentient_util.constants import EPS
from constants import DEVICE
from enums import ActionSpace
from mean_reward import MeanReward
from networks.fully_connected import FullyConnected
from rewards_csv import EvaluateRewardsCsv
from torch import nn
from torch import nn as nn
from torch import tensor
from torch.distributions import Categorical
from torch.nn import functional as F
from torch.optim import Adam


@dataclass
class SavedAction:
    log_prob: float
    value: float


class Actor(nn.Module):
    """"""

    def __init__(self, *, config, state_size: int, action_size: int):

        super().__init__()

        model_config = config.agent.torch_models.main

        self.model = FullyConnected(
            input_size=state_size,
            hidden_dims=model_config.hidden_dims,
            output_size=model_config.hidden_dims[-1],
            activations=model_config.activations,
        )

        # actor's layer
        self.action_head = nn.Linear(model_config.hidden_dims[-1], action_size)

        # critic's layer
        self.value_head = nn.Linear(model_config.hidden_dims[-1], 1)

        self.to(DEVICE)

    def forward(self, state):

        x = self.model(state)

        action_scores = self.action_head(x)
        state_values = self.value_head(x)

        return F.softmax(action_scores, dim=-1), state_values


class ActorCriticAgent(AbstractAgent):
    """"""

    def __init__(self, *, config, env):

        super().__init__(config=config, env=env)

        state_size = self.env.state_size  # type: ignore
        action_size = self.env.action_space.n  # type: ignore

        self.actor = Actor(
            config=config, state_size=state_size, action_size=action_size
        )
        self.actor_optimizer = Adam(self.actor.parameters(), lr=config.agent.actor_lr)

        self.load()

        if config.agent.training:
            self.actor.train()
        else:
            self.actor.eval()

        # Define buffer space for policy retraining at end of episode
        self.saved_actions: deque[SavedAction] = deque()
        self.rewards: deque[float] = deque()

        self.network_save_counter = 0

    def __enter__(self):
        self.rewards_csv = EvaluateRewardsCsv(config=self.config)
        return self

    def __exit__(self, type, value, traceback):
        self.rewards_csv.close()
        if type or value or traceback:
            logging.error(f"__exit__ traceback: {type=}, {value=}, {traceback=}")

    def select_action(self, state):
        """Select action from a Categorical distribution."""

        probs, state_value = self.actor(state)

        m = Categorical(probs)

        # ToDo: turn back to this if the AC ever needs to become operational
        # action = m.sample() if self.config.agent.sample else torch.argmax(probs)

        action = torch.argmax(probs)

        self.saved_actions.append(SavedAction(m.log_prob(action), state_value))
        return action.item()

    def __call__(self):

        config = self.config

        @contextmanager
        def EndStep(*, rewards_csv):

            try:
                yield

            finally:

                mean_reward(reward)

                try:
                    # Only applies to Sentient envs
                    rewards_csv(
                        episode=episode,
                        step=step_info["episode_step"],
                        reward=reward,
                        it=step_info["it"],
                        fill_size=step_info["fill_size"],
                        pf=step_info["pf"],
                        action=action,
                        state=state,
                        order_depth=step_info["order_depth"],
                    )

                except KeyError:
                    pass

        with MeanReward(config) as mean_reward:

            for episode in range(config.agent.episodes):

                episode_reward = 0

                state = tensor(self.env.reset())

                done = False
                while not done:

                    with EndStep(rewards_csv=config.rewards_csv):  # type: ignore

                        action = self.select_action(state)

                        next_state, reward, done, step_info = self.env.step(
                            action=action  # type: ignore
                        )  # type: ignore
                        episode_reward += reward

                        next_state = tensor(next_state)

                        self.rewards.append(reward)

                        self.decay_epsilon()

                        state = next_state

                else:

                    self.log_episode(episode, mean_reward())

                    self.finish_episode(episode=episode)

                    if self.child_process_required(episode):
                        self.launch_child_process()

            else:
                if config.agent.training and not self.pytest:
                    self.save()

                self.env.close()

    def finish_episode(self, *, episode):

        config = self.config

        R = 0
        policy_losses = []
        value_losses = []
        bellman_rewards = []

        for i, r in enumerate(reversed(self.rewards)):

            R = r + config.agent.gamma * R
            bellman_rewards.insert(0, R)

        bellman_rewards = tensor(bellman_rewards)

        rewards = bellman_rewards

        # Normalize
        rewards = (rewards - rewards.mean()) / (rewards.std() + EPS)

        for saved_action, r in zip(self.saved_actions, rewards):
            reward = r - saved_action.value
            policy_losses.append(-saved_action.log_prob * reward)

            value_losses.append(
                F.smooth_l1_loss(
                    tensor(saved_action.value),
                    tensor([r]),
                )
            )

        self.actor_optimizer.zero_grad()

        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

        loss.backward()

        self.actor_optimizer.step()

        self.saved_actions.clear()
        self.rewards.clear()
