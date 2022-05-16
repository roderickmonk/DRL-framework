#!/usr/bin/env python
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import importlib
import os
import random
from typing import Any

import gym
import numpy as np
import torch
from sentient_util import cfg, logger
from torch import cuda


def test_qrdqn():

    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)

    config = {
        "instance_id": 0,
        "agent": {
            "agent": "dqn.qrdqn_agent",
            "torch_models": {
                "main": {"hidden_dims": [128], "activations": ["nn.Tanh"],}
            },
            "buffer_size": 10000,
            "batch_size": 32,
            "episodes": 1000,
            "actor_lr": 1e-3,
            "quantile_resolution": 2,
            "gamma": 0.99,
            "epsilon_decay": {
                "type": "EXPONENTIAL",
                "start": 1.0,
                "end": 0.1,
                "rate": 0.998,
            },
            "network_nodes": 256,
            "nn_trace": None,
            "target_update_interval": 100,
            "training": True,
            "training_interval": 1,
            "purge_network": True,
            "network": None,
        },
        "env_name": "MountainCar-v0",
        "misc": {
            "seed": 7,
            "generate_csv": False,
            "log_interval": 10,
            "purge_network": True,
            "log_level": "CRITICAL",
        },
    }

    logger.setup()

    config: Any = cfg.ConfigObj(config)

    agent_module = importlib.import_module(config.agent.agent)
    agent = getattr(agent_module, "Agent")

    if bool(os.getenv("CO3_FORCE_PYTEST", None)) or cuda.is_available():
        assert (
            agent(config=config, env=gym.make(config.env_name))() is True
        ), "Agent Unable to Learn"
