import itertools as it
import logging
import os
import random
import sys
from abc import ABC, abstractmethod
from contextlib import contextmanager
from copy import copy, deepcopy
from dataclasses import dataclass
from typing import Any

import constants
import gym
import numpy as np
import torch
import util
from constants import DEVICE, noop
from decorators import call_counter
from env_modules.env import Env
from project_functions import float_tensor
from pydantic import NoneIsAllowedError
from pydantic_config import *
from replay_buffer import ReplayBuffer
from sentient_util import cfg, logger
from torch import Tensor, nn
from wrappers import ActionWrapper
from devtools import debug

ProcessConfigs = (
    PpoProcessConfig
    | DdpgProcessConfig
    | SacProcessConfig
    | Td3ProcessConfig
    | DDPG_SimEnv1_Config
    | TD3_SimEnv1_Config
)


@dataclass(eq=False, order=False)
class AbstractAgent(ABC):

    config: ProcessConfigs
    env: Env
    network_path: Any = None
    pytest = "pytest" in sys.modules

    def __post_init__(self):

        super().__init__()

        self.actor: nn.Module

        config = self.config

        logger.set_log_level(level=config.misc.log_level)

        if constants.CONSOLE != "logging":
            logging.disable()

        if self.pytest:
            torch.use_deterministic_algorithms(True)

        random.seed(config.misc.seed)
        np.random.seed(config.misc.seed)
        torch.manual_seed(config.misc.seed)
        self.env.seed(config.misc.seed)

        self.gamma = getattr(config.agent, "gamma", None)

        self.target_update_interval: int = config.agent.target_update_interval

        self.tau = getattr(config.agent, "tau", 0.001)

        self.training = config.agent.training
        self.purge_network = config.agent.purge_network

        self.training_counter = it.count(0)

        self.set_network_path()

        try:
            logging.debug(f"{AbstractAgent.network_path=}")
        except AttributeError:
            pass

        try:
            self.eps: float = (
                config.agent.epsilon_decay.start
                if hasattr(config.agent, "epsilon_decay")
                and config.agent.epsilon_decay is not None
                else 0.0
            )
        except (AttributeError, TypeError):
            self.eps: float

        self.leave_progress_bar = config.misc.leave_progress_bar

        self.pytest_success_threshold: float
        self.replay_buffer: ReplayBuffer

        self.components = set(
            [
                "actor",
                "actor_optimizer",
                "actor_target",
                "critic",
                "critic_optimizer",
                "critic_target",
            ]
        )

        self.pytest_success_threshold = -150

        logging.debug("Agent Initialized")

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        pass

    def set_network_path(self):
        """"""

        if self.training:

            config = self.config

            AbstractAgent.network_path = util.set_path(
                configured_path=config.agent.network
                if hasattr(config.agent, "network")
                else None,
                nominal_folder="networks",
                nominal_ext=".pt",
            )

            if self.purge_network:
                try:
                    os.remove(AbstractAgent.network_path)
                except (OSError, IOError):
                    pass

    def child_process_required(self, episode) -> bool:

        config = self.config

        if (child_process := getattr(config, "child_process", None)) is not None:

            delay = child_process.launch_delay

            return (
                episode >= delay
                and (episode - delay) % child_process.launch_interval == 0
            )

        else:
            return False

    def launch_child_process(self) -> float:

        parent_config = self.config

        logging.critical(f"Launching Child Process")

        # Save the network so the child can infer from it
        self.save()

        # child will take on many of the config params from the parent
        child_config = deepcopy(parent_config)

        # The child process does not beget children
        child_config.child_process = None

        # Confine child process to testing
        child_config.agent.training = False

        def merge(source: Any, to: Any, attr: str):
            if (x := getattr(source, attr)) is not None:
                setattr(to, attr, x)

        """   Overwrite a select set of attributes with the child's  """
        if source := getattr(parent_config.child_process, "misc"):
            to = child_config.misc
            merge(source, to, "csv_path")
            merge(source, to, "generate_csv")
            merge(source, to, "log_interval")

        if source := getattr(parent_config.child_process, "agent"):
            to = child_config.agent
            merge(source, to, "buffer_size")
            merge(source, to, "batch_size")
            merge(source, to, "episodes")

        # Some environments need to know the child process episodes
        self.env.env_config.episodes = child_config.agent.episodes

        if source := getattr(parent_config.child_process, "env_config"):
            to = child_config.env_config
            merge(source, to, "datasets")
            merge(source, to, "randomize_dataset_reads")

        from playback import Playback

        # Set the env into testing and startup the child process
        with self.env.env2test(), Playback(
            config=child_config, env=self.env
        ) as playback:
            MR = playback()
            logging.info("Resume Parent Process")
            return MR

    def log_episode(self, episode, MR):

        if (
            episode != 0
            and self.config.agent.episodes is not None
            and (
                episode % self.config.misc.log_interval == 0
                or episode == self.config.agent.episodes - 1
            )
        ) and not self.pytest:
            logging.info(
                f"Episode: {episode:5}, "
                f"Mean Reward: {MR:10.6f} "
                f"{util.log_interval()}"
            )

    @call_counter
    def decay_epsilon(self):

        if self.eps and self.config.agent.epsilon_decay:

            epsilon_decay = self.config.agent.epsilon_decay

            if epsilon_decay.type == "LINEAR":

                self.eps = max(
                    epsilon_decay.end,
                    epsilon_decay.rate * self.eps,
                )

            elif epsilon_decay.type == "EXPONENTIAL":

                self.eps = epsilon_decay.end + (
                    epsilon_decay.start - epsilon_decay.end
                ) * np.exp(-self.decay_epsilon.count * (1 - epsilon_decay.rate))

            else:
                raise ValueError("Unknown Epsilon Decay Type")

    @abstractmethod
    def __call__(self):
        pass

    @abstractmethod
    def select_action(self, state) -> Tensor:
        pass

    @abstractmethod
    def train(self, *args):
        pass

    def test(self, *args):
        # Do nothing more than sample from the replay buffer
        self.replay_buffer.sample()

    @abstractmethod
    def predicted(self, *args: Any):
        pass

    def env_state(self):

        # Can only return something if env is Sentient
        try:
            return {
                "DF_index": self.env.DF_index,  # type:ignore
                "current_dataset": self.env.current_dataset,  # type:ignore
                "episode_step": 1,  # self.env.episode_step,
                "current_step": self.env.current_step,
            }
        except AttributeError:
            return {}

    def load_env_state(self, checkpoint: dict):

        if (
            "env_state" in checkpoint
            and bool(checkpoint["env_state"])
            and self.config.agent.training
        ):

            env_state: Any = cfg.ConfigObj(checkpoint["env_state"])
            self.env.episode_step = 1
            self.env.current_step = env_state.current_step

            self.env.DF_index = env_state.DF_index  # type:ignore
            self.env.current_dataset = env_state.current_dataset  # type:ignore

    def pytest_successful2(self, steps_done, episode, episode_reward):

        if self.pytest:

            if (
                episode % self.config.misc.log_interval == 0
                or episode_reward > self.pytest_success_threshold
            ):

                logging.critical(
                    f"episode={episode:4}, "
                    f"steps={steps_done:6}, "
                    f"{episode_reward=}"
                )

            # Learning is proven by the following check
            if episode_reward > self.pytest_success_threshold:
                return True

        return False

    def pytest_successful(self, episode, eval_interval=10, eval_episodes=10):

        if all([self.pytest, episode, episode % eval_interval]) == 0:

            with torch.no_grad():

                eval_rewards = 0

                for _ in range(eval_episodes):

                    state = float_tensor(self.env.reset()).squeeze()

                    done = False
                    while not done:
                        action = self.select_action(state)
                        state, reward, done, _ = self.env.step(  # type:ignore
                            action.detach().cpu().numpy()
                        )
                        state = float_tensor(state).squeeze()
                        eval_rewards += reward.item()

                        logging.debug(f"{eval_rewards=}, {reward=}")

                else:

                    eval_rewards /= eval_episodes

                    logging.critical(f"episode={episode:4}, {eval_rewards=:8.1f}")

                    if self.test_evaluation(eval_rewards):
                        logging.critical(f"Environment Solved")
                        return True

                logging.debug(f"{eval_rewards=}")

        return False

    def save(self):

        if hasattr(AbstractAgent, "network_path"):

            save = {
                "eps": self.eps if hasattr(self, "eps") else None,
                "random_states": self.random_states(),
                "env_state": self.env_state(),
            }

            for x in self.components:
                if hasattr(self, x):
                    save[x] = getattr(self, x).state_dict()

            torch.save(save, AbstractAgent.network_path)

            logging.error(f"PyTorch Save: {AbstractAgent.network_path}")

            if self.config.agent.save_onnx:

                onnx_file = f"/onnx/{self.config.misc.market}.onnx"
                try:
                    torch.onnx.export(
                        self.actor,
                        torch.randn(
                            # provide a sample input
                            # (1, *self.env.observation_space.shape),  # type:ignore
                            (1, 2, 40),  # type:ignore
                            device=DEVICE,
                        ),
                        onnx_file,
                    )
                except (PermissionError, RuntimeError) as err:
                    logging.error(f"Unable to export {onnx_file}, Error: {err}")

    def load(self):

        if not hasattr(AbstractAgent, "network_path") or not os.path.exists(
            AbstractAgent.network_path
        ):
            return None

        checkpoint = torch.load(AbstractAgent.network_path)

        if self.training and "eps" in checkpoint:
            self.eps = checkpoint["eps"]

        if "random_states" in checkpoint:
            self.random_states(checkpoint["random_states"])

        self.load_env_state(checkpoint)

        for x in self.components:
            if self.training or x in ["actor", "critic"]:
                if hasattr(self, x):
                    logging.debug(f"PyTorch Loading of {x=}")
                    getattr(self, x).load_state_dict(checkpoint[x])

        logging.debug(f"PyTorch Load: {AbstractAgent.network_path}")

        return checkpoint

    def test_evaluation(self, evaluation_rewards):

        try:
            if (
                evaluation_rewards > self.env.spec.reward_threshold * 1.1  # type:ignore
            ):  # x 1.1 because of small eval_episodes
                return True

        except TypeError:
            if evaluation_rewards > self.pytest_success_threshold:
                logging.critical(
                    f"{evaluation_rewards=}, {self.pytest_success_threshold=}"
                )
                return True

        return False

    @contextmanager
    def evayml_env(self):
        class ActionWrapper(gym.ActionWrapper):
            def __init__(self, env):
                super().__init__(env)

            def action(self, action: float):
                return 2 * action

        try:
            env = gym.make(self.config.env_config.env_name)
            env = ActionWrapper(env)

            env.seed(self.config.misc.seed + 100)
            yield env
        finally:
            env.close()  # type:ignore

    @staticmethod
    def random_states(random_states=None):

        if random_states is None:
            return {
                "random_state": random.getstate(),
                "np_random_state": np.random.get_state(),
                "pt_random_state": torch.random.get_rng_state(),
            }
        else:
            random.setstate(random_states["random_state"])
            np.random.set_state(random_states["np_random_state"])
            torch.random.set_rng_state(random_states["pt_random_state"])
            logging.debug(f"Random Generator States Reloaded")
            return None

    @staticmethod
    def hard_update(target, source):
        with torch.no_grad():
            for t, s in zip(target.parameters(), source.parameters()):
                t.data.copy_(s.data)

    def soft_update(self, target, source):

        with torch.no_grad():
            for t, s in zip(target.parameters(), source.parameters()):
                t.data.copy_(t.data * (1.0 - self.tau) + s.data * self.tau)
