#!/usr/bin/env python
import itertools as it
from collections import deque
from dataclasses import dataclass
from itertools import count
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from pydantic_config import ProcessConfig
from torch.utils.tensorboard.writer import SummaryWriter

from sentient_tensorboard.plot import Plot

training_episodes = 0
training_episodes_counter = count(0)


def increment_training_episodes():
    global training_episodes
    training_episodes = next(training_episodes_counter)


class PlotReward(Plot):
    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        if (xx := getattr(self, "training", None)) is not None and not xx:

            self.active = True
            self.rewards = deque()

    def __call__(self, **kwargs):

        self.active and self.rewards.append(kwargs["reward"])

    def plot(self):

        if self.active:

            self.fig = plt.figure(figsize=(18, 10), dpi=80)

            ax = plt.axes()
            ax.set_facecolor("whitesmoke")

            plt.xlabel("Episode")
            plt.title(f"Reward After {training_episodes} Training Episodes")
            plt.ylabel("Reward")

            plt.plot(
                np.arange(len(self.rewards)),
                np.array(self.rewards),
                linewidth=1.5,
                c="b",
            )

            graph_name = f"rewards-child-process-{self.instance_id}.png"  # type:ignore

            self.writer.add_figure(
                graph_name,
                self.fig,
                global_step=None,
                close=True,
                walltime=None,
            )

            plt.close()
