import importlib
import itertools as it
import logging
import os
import random

import gym
import numpy as np
import torch
from pydantic_config import *

from wrappers import ActionWrapper

_log = logging.warning


class Playback:
    """"""

    instance_id = it.count(0)
    env: gym.Env | None = None

    def __init__(self, *, config, child_datasets=None, env: gym.Env):

        if child_datasets is not None:
            global datasets
            datasets = [child_datasets]

        self.config = config
        self.env = env

        agent_module = importlib.import_module(config.agent.agent)
        self.agent = getattr(agent_module, "Agent")

    def __enter__(self):

        # Record the state of all random number generators
        self.py_random_state = random.getstate()
        self.np_random_state = np.random.get_state()
        self.pt_random_state = torch.random.get_rng_state()

        return self

    def __exit__(self, *args):

        # Reinstate all random number generator states
        random.setstate(self.py_random_state)
        np.random.set_state(self.np_random_state)
        torch.random.set_rng_state(self.pt_random_state)

    def __call__(self) -> float:

        config = self.config

        setattr(config, "instance_id", next(Playback.instance_id))

        with self.agent(config=self.config, env=self.env) as agent:
            _log(f"Process Instance {self.config.instance_id}")
            MR = agent()

        return MR
