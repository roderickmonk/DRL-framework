import itertools as it
import logging
import os

import constants
import numpy as np
import torch
from agent.agent import AbstractAgent
from constants import DEVICE, Experience
from mean_reward import MeanReward
from network_trace import NetworkTrace
from torch import tensor
from torch.optim import Adam
from tqdm.auto import trange

torch.set_printoptions(edgeitems=100, linewidth=1000)


class DqnBase(AbstractAgent):
    """"""

    def __init__(self, network, *, config, env):

        super().__init__(config=config, env=env)

        self.network_trace = NetworkTrace(self.config)

        network_kwargs = dict(
            state_size=self.env.observation_space.shape[0],  # type: ignore
            action_size=self.env.action_space.n,  # type: ignore
            config=config,
        )

        self.actor = network(**network_kwargs).to(DEVICE)

        self.tau: torch.Tensor | float

        if "nn_trace" in network_kwargs:
            network_kwargs["nn_trace"] = None

        if self.training:

            self.actor_target = network(**network_kwargs).to(DEVICE)

            for network_tensor in self.actor.state_dict():
                logging.debug(
                    f"{network_tensor=}, size={self.actor.state_dict()[network_tensor].size()}"
                )

            self.actor_optimizer = Adam(
                self.actor.parameters(), lr=self.config.agent.actor_lr
            )

        self.load()

        logging.debug("DqnBase Initialized")

    def env_state(self):
        return {
            "episode_step": 1,
            "current_step": self.env.current_step,
        }

    def __call__(self):
        """"""

        training_required = (
            lambda steps_done: config.agent.training
            and steps_done % config.agent.training_interval == 0
            and len(self.replay_buffer) >= batch_size
        )

        def csv_write():

            try:
                # Only applies to Sentient envs
                config.rewards_csv(  # type: ignore
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
            except (KeyError, AttributeError):
                pass

        batch_size, config = self.config.agent.batch_size, self.config  # type: ignore

        _steps_done = it.count(-1)
        steps_done = next(_steps_done)

        with MeanReward(config) as mean_reward:

            with trange(
                config.agent.episodes,
                disable=constants.CONSOLE != "progress_bar",
                colour="blue" if config.child_process else "green",
                leave=self.leave_progress_bar,
            ) as t:

                for episode in t:

                    episode_reward = 0

                    state = tensor(self.env.reset())

                    done = False
                    while not done:

                        action = self.select_action(state=state)

                        next_state, reward, done, step_info = self.env.step(
                            action=action  # type: ignore
                        )
                        episode_reward += reward

                        next_state = tensor(next_state)

                        self.replay_buffer + Experience(
                            state=state,
                            action=tensor([action]),
                            next_state=next_state,
                            reward=tensor([reward]),
                            done=tensor([done]),
                        )

                        steps_done = next(_steps_done)

                        if training_required(steps_done):
                            self.train(steps_done)

                        mean_reward(reward)

                        csv_write()

                        self.decay_epsilon()

                        state = next_state

                    else:

                        self.log_episode(episode, mean_reward())

                        if self.pytest_successful2(steps_done, episode, episode_reward):
                            return True

                        if self.child_process_required(episode):
                            self.launch_child_process()

                    t.set_description(
                        f"{config.instance_id:>3}  MR={mean_reward():>10.4e}  "
                    )
                else:
                    if config.agent.training and not self.pytest:
                        self.save()

                    self.env.close()
