#!/usr/bin/env python

import warnings

warnings.filterwarnings("ignore", category=UserWarning)
import importlib
import os

import gym
import numpy as np
from sentient_util import logger
from pydantic_config import Td3PytestConfig
from wrappers import ActionWrapper


def test_td3_with_pendulum():

    config = {
        "instance_id": 0,
        "agent": {
            "torch_models": {
                "Actor": {"hidden_dims": [256, 128], "activations": ["nn.ReLU"],},
                "Critic": {"hidden_dims": [256, 128], "activations": ["nn.ReLU"],},
            },
            "gradient_clipping": {"max_norm": 1.0, "norm_type": 2.0},
            "exploration": 100,
            "episodes": 20000,
            "gamma": 0.99,
            "actor_lr": 3e-4,
            "critic_lr": 3e-4,
            "tau": 0.005,
            "gamma": 0.99,
            "policy_noise": 0.2,
            "noise_clip": 0.5,
            "batch_size": 64,
            "buffer_size": 100000,
            "training": True,
            "training_interval": 2,
            "action_noise": {"type": "OrnsteinUhlenbeck", "sigma": 0.02},
        },
        "env_name": "Pendulum-v1",
        "misc": {
            "log_level": "CRITICAL",
            "log_interval": 10,
            "seed": 10,
            "log_level": "INFO",
            "generate_csv": False,
            "csv_path": None,
            "record_state": False,
        },
    }

    logger.setup()

    config = Td3PytestConfig(**config)

    agent_module = importlib.import_module(config.agent.agent)
    agent = getattr(agent_module, "Agent")

    assert (
        config.env_name is not None
        and agent(config=config, env=ActionWrapper(gym.make(config.env_name)))() is True
    ), "Agent Unable to Learn"
