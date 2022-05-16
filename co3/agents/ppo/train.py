import os
from datetime import datetime

import gym
import numpy as np
import torch
from agents.ppo.ppo_agent import Agent
from omegaconf import OmegaConf
import pytest


def train(self, *args):

    env = gym.make(env_name := "Pendulum-v1")

    print("training environment name : " + env_name)

    config = OmegaConf.create(
        {
            "agent": {
                "actor_lr": 0.0003,
                "critic_lr": 0.001,
                "gamma": 0.99,
                "K_epochs": 80,
                "eps_clip": 0.2,
                "exploration": 0,
                "batch_size": 64,
                "buffer_size": 1000,
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

    # initialize a PPO agent
    agent = Agent(config=config, env=env)

    time_step = 0

    for episode in range(config.agent.episodes):

        state = env.reset()
        episode_reward = 0

        done = False
        while not done:

            # select action with policy
            action = agent.select_action(state)
            state, reward, done, _ = env.step(action)

            # saving reward and is_terminals
            agent.buffer.rewards.append(reward)
            agent.buffer.is_terminals.append(done)

            time_step += 1
            episode_reward += reward

            # if continuous action space; then decay action std of ouput action distribution
            if time_step % config.agent.action_std_decay_freq == 0:
                agent.decay_action_std(
                    config.agent.action_std_decay_rate, config.agent.min_action_std
                )

            if done:
                if episode != 0 and episode % 5 == 0:
                    agent.train()

                if episode != 0 and episode % config.misc.log_interval == 0:
                    print(f"{episode=}, {episode_reward=}")

                break

    env.close()


if __name__ == "__main__":

    train()
