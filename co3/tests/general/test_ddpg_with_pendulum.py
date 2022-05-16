#!/usr/bin/env python

import importlib
import logging
import random
import warnings
from typing import Any
from devtools import debug

warnings.filterwarnings("ignore", category=UserWarning)
import os

import constants
import gym
from pydantic_config import DdpgPytestConfig
from sentient_util import logger
from wrappers import ActionWrapper


def test_ddpg_agent():

    constants.CONSOLE = "logging"

    config = {
        "agent": {
            "torch_models": {
                "Actor": {
                    "hidden_dims": [256, 128],
                    "activations": ["nn.ReLU"],
                },
                "Critic": {
                    "hidden_dims": [256, 128],
                    "activations": ["nn.ReLU"],
                },
            },
            "episodes": 100000,
            "exploration": 100,
            "training_interval": 2,
            "batch_size": 64,
            "buffer_size": 100000,
            "actor_lr": 0.0001,
            "critic_lr": 0.001,
            "gamma": 0.99,
            "tau": 0.005,
        },
        "misc": {
            "log_level": "CRITICAL",
            "log_interval": 10,
            "seed": 10,
        },
        "env_name": "Pendulum-v1",
    }

    logger.setup()

    config = DdpgPytestConfig(**config)

    debug(config)

    breakpoint()

    agent = getattr(importlib.import_module(config.agent.agent), "Agent")

    assert (
        config.env_name is not None
        # and agent(config=config, env=ActionWrapper(gym.make(config.env_name)))() is True
        and agent(config=config, env=gym.make(config.env_name))() is True
    ), "Agent Unable to Learn"


if __name__ == "__main__":

    test_ddpg_agent()
