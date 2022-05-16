#!/usr/bin/env python

import warnings

warnings.filterwarnings("ignore", category=UserWarning)
import importlib
import os

import gym
from pydantic_config import SacPytestConfig
from sentient_util import logger
from wrappers import ActionWrapper


def test_sac_agent():

    config = {
        "agent": {
            "torch_models": {
                "Critic": {"hidden_dims": [256, 128, 64], "activations": ["nn.ReLU"],},
                "Actor": {"hidden_dims": [256, 128, 64], "activations": ["nn.ReLU"],},
            },
            "gradient_clipping": {"max_norm": 1.0, "norm_type": 2.0},
            "gamma": 0.99,
            "tau": 0.005,
            "actor_lr": 0.0003,
            "critic_lr": 0.0003,
            "alpha": 0.2,
            "automatic_entropy_tuning": True,
            "episodes": 1000,
            "exploration": 100,
            "batch_size": 64,
            "buffer_size": 100000,
            "training": True,
            "training_interval": 2,
            "action_noise": None,
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

    config = SacPytestConfig(**config)

    agent_module = importlib.import_module(config.agent.agent)
    agent = getattr(agent_module, "Agent")

    assert (
        config.env_name is not None
        and agent(config=config, env=ActionWrapper(gym.make(config.env_name)))() is True
    ), "Agent Unable to Learn"
