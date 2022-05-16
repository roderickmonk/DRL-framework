"""
This code is taken from stable-baselines https://github.com/openai/baselines
"""

import numpy as np
import torch
from torch import tensor

from constants import DEVICE


class NoNoise(object):
    def __call__(self):
        return 0.0

    def reset(self):
        pass


class NormalActionNoise(NoNoise):
    """
    A Gaussian action noise
    :param mean: (float) the mean value of the noise
    :param sigma: (float) the scale of the noise (std here)
    """

    def __init__(self, *, mean, sigma, size):
        self._size = size
        self._mu = mean
        self._sigma = sigma

    def __call__(self):
        return tensor(
            np.random.normal(self._mu, self._sigma, size=self._size), torch.float32
        )

    def __repr__(self):
        return "NormalActionNoise(mu={}, sigma={})".format(self._mu, self._sigma)


class OrnsteinUhlenbeckActionNoise(NoNoise):
    """
    A Ornstein Uhlenbeck action noise, this is designed to approximate brownian motion with friction.
    Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
    :param mean: (float) the mean of the noise
    :param sigma: (float) the scale of the noise
    :param theta: (float) the rate of mean reversion
    :param dt: (float) the timestep for the noise
    :param initial_noise: ([float]) the initial value for the noise output, (if None: 0)
    """

    def __init__(self, *, mean, sigma=0.2, theta=0.15, dt=1e-2, initial_noise=None):
        self._theta = theta
        self._mu = mean
        self._sigma = sigma
        self._dt = dt
        self.initial_noise = initial_noise
        self.noise_prev: float
        self.reset()

    def __call__(self):
        noise = (
            self.noise_prev
            + self._theta * (self._mu - self.noise_prev) * self._dt
            + self._sigma * np.sqrt(self._dt) * np.random.normal(size=self._mu.shape)
        )
        self.noise_prev = noise
        return tensor(noise, device=DEVICE)

    def reset(self):
        """
        reset the Ornstein Uhlenbeck noise, to the initial position
        """
        self.noise_prev = (
            self.initial_noise
            if self.initial_noise is not None
            else np.zeros_like(self._mu)
        )

    def __repr__(self):
        return f"OrnsteinUhlenbeckActionNoise, mu={self._mu}, sigma={self._sigma}"


def set_action_noise(
    *, action_size, config
) -> NoNoise | NormalActionNoise | OrnsteinUhlenbeckActionNoise:

    if (
        not config.agent.training
        or "action_noise" not in config.agent
        or config.agent.action_noise is None
    ):
        return NoNoise()

    if config.agent.action_noise.type not in [None, "Gaussian", "OrnsteinUhlenbeck"]:
        raise ValueError("Invalid action_noise: {config.action_noise}")

    if config.agent.action_noise.type == "Gaussian":
        return NormalActionNoise(
            mean=0, sigma=config.agent.action_noise.sigma, size=action_size
        )
    elif config.agent.action_noise.type == "OrnsteinUhlenbeck":
        return OrnsteinUhlenbeckActionNoise(
            mean=np.zeros(action_size), sigma=config.agent.action_noise.sigma
        )
    else:
        return NoNoise()
