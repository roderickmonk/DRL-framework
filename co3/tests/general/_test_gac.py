#!/usr/bin/env python

import logging
import os
import random
import warnings
from typing import Any

warnings.filterwarnings("ignore", category=UserWarning)
import importlib

import gym
import numpy as np
import pytest
import torch
from sentient_util import cfg, logger


def test_gac_agent():

    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)

    config = {
        "instance_id": 0,
        "agent": {
            "agent": "gac.gac_agent",
            "torch_models": {
                "Critic": {"hidden_dims": [400, 300], "activations": ["nn.ReLU"],},
            },
            "actor_lr": 1e-4,
            "critic_lr": 1e-3,
            "value_lr": 1e-3,
            "layer1_size": 400,
            "layer2_size": 300,
            "network_nodes": 256,
            "purge_network": True,
            "network": None,
            "target_update_interval": 1,
            "training": True,
            "gamma": 0.99,
            "tau": 0.01,
            "batch_size": 128,
            "episodes": 100000,
            "exploration": 10000,
            "training_interval": 500,
            "buffer_size": 200000,
            "training_actor_samples": 32,
            "not_autoregressive": False,
            "q_normalization": 0.01,
            "target_policy": "exponential",
            "target_policy_q": "min",
            "boltzman_temperature": 1.0,
            "action_noise": {"type": "OrnsteinUhlenbeck", "sigma": 0.02},
        },
        "misc": {
            "seed": 10,
            "log_interval": 1,
            "generate_csv": False,
            "log_level": "CRITICAL",
        },
        "env_name": "Pendulum-v1",
    }

    config: Any = cfg.ConfigObj(config)
    logger.setup()

    agent_module = importlib.import_module(config.agent.agent)
    agent = getattr(agent_module, "Agent")

    if bool(os.getenv("CO3_FORCE_PYTEST", None)) or torch.cuda.is_available():
        assert agent(config=config)() is True, "Agent Unable to Learn"

