#!/usr/bin/env python

import warnings

warnings.filterwarnings("ignore", category=UserWarning)
import importlib

import gym
from omegaconf import OmegaConf
from sentient_util import cfg, logger
from pydantic_config import PpoProcessConfig


def test_ppo_agent():

    config = OmegaConf.create(
        {
            "instance_id": 0,
            "agent": {
                "agent": "ppo.ppo_agent",
                "actor_lr": 0.0003,
                "critic_lr": 0.001,
                "gamma": 0.99,
                "K_epochs": 80,
                "eps_clip": 0.2,
                "network": None,
                "episodes": 10000,
                "training_interval": 5,
                "target_update_interval": 2,
                "action_std_decay_freq": 2.5e5,
                "action_std_init": 0.6,
                # linearly decay action_std (action_std = action_std - action_std_decay_rate)
                "action_std_decay_rate": 0.05,
                # minimum action_std (stop decay after action_std <= min_action_std)
                "min_action_std": 0.1,
            },
            "env_name": "Pendulum-v1",
            "misc": {
                "log_level": "INFO",
                "generate_csv": False,
                "seed": 7,
                "log_interval": 10,
            },
        }
    )

    config = PpoProcessConfig(**config)

    from devtools import debug as db

    db(config)

    logger.setup()
    logger.set_log_level("INFO")

    agent_module = importlib.import_module(config.agent.agent)
    agent = getattr(agent_module, "Agent")

    assert (
        config.env_name
        and agent(config=config, env=gym.make(config.env_name))() is True
    ), "Agent Unable to Learn"
