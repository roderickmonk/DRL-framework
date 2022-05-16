#!/usr/bin/env python

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import importlib
import logging
import os

import gym
from sentient_util import logger
import torch
from constants import DEVICE
from torch import tensor
from agent.agent import AbstractAgent
from pydantic_config import SacPytestConfig
from wrappers import ActionWrapper


def pytest_successful(self, episode, eval_interval=10, eval_episodes=10):

    if all([self.pytest, episode, episode % eval_interval]) == 0:

        with torch.no_grad():

            eval_rewards = 0

            for _ in range(eval_episodes):

                state = tensor(self.env.reset(), device=DEVICE)

                done = False
                steps = 0
                while not done:

                    steps += 1

                    action = self.select_action(state)
                    state, reward, done, _ = self.env.step(
                        action.detach().cpu().numpy()
                    )
                    state = tensor(state, device=DEVICE)
                    eval_rewards += reward

            else:

                eval_rewards /= eval_episodes

                logging.critical(f"episode={episode:4}, {eval_rewards=:8.1f}")

                if self.test_evaluation(eval_rewards):
                    logging.critical(f"Environment Solved")
                    return True

    return False


def test_sac_with_copter():

    config = {
        "agent": {
            "torch_models": {
                "Actor": {"hidden_dims": [256, 256], "activations": ["nn.ReLU"],},
                "Critic": {"hidden_dims": [256, 256], "activations": ["nn.ReLU"],},
            },
            "gradient_clipping": {"max_norm": 1.0, "norm_type": 2.0},
            "exploration": 125,
            "episodes": 500,
            "gamma": 0.99,
            "actor_lr": 3e-4,
            "critic_lr": 3e-4,
            "tau": 0.005,
            "gamma": 0.99,
            "policy_noise": 0.2,
            "noise_clip": 0.5,
            "batch_size": 256,
            "buffer_size": 1000000,
            "training": True,
            "training_interval": 1,
            "action_noise": {"type": "OrnsteinUhlenbeck", "sigma": 0.02},
        },
        "env_name": "gym_copter:Lander2D-v0",
        "misc": {
            "log_level": "ERROR",
            "log_interval": 1,
            "seed": 10,
            "log_level": "INFO",
            "generate_csv": False,
            "csv_path": None,
            "record_state": False,
        },
    }

    from pyvirtualdisplay import Display

    display = Display(visible=False, size=(400, 300))
    display.start()

    logger.setup()

    config = SacPytestConfig(**config)

    agent_module = importlib.import_module(config.agent.agent)
    agent = getattr(agent_module, "Agent")

    assert config.env_name
    agent = agent(config=config, env=ActionWrapper(gym.make(config.env_name)))
    agent.pytest_success_threshold = 175
    setattr(AbstractAgent, "pytest_successful", pytest_successful)
    assert agent() is True, "Agent Unable to Learn"
