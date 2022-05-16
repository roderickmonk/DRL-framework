import logging
import os
from collections import deque
from typing import Tuple
from torch import Tensor
import numpy as np
import torch
from constants import DEVICE, Experience
from sentient_util.exceptions import InvalidConfiguration


try:
    profile  # type: ignore
except NameError:
    profile = lambda x: x


class ReplayBuffer:
    def __init__(self, *, batch_size: int, buffer_size: int):

        if batch_size > buffer_size:
            raise InvalidConfiguration(f"{batch_size=} > {buffer_size=}")

        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.deque = deque(maxlen=buffer_size)
        self.range = np.random.default_rng()  # type: ignore

    def __add__(self, e: Experience):
        # print (f"{e.state.shape=}, {e.action.shape=}")
        self.deque.append(e)

    @profile  # type:ignore
    def sample(self) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:

        if self.batch_size < self.buffer_size:

            indices = self.range.integers(low=0, high=len(self), size=self.batch_size)
            batches = zip(*[self.deque[idx] for idx in indices])

        else:
            # return the entire buffer without randomization
            batches = zip(*[x for x in self.deque])

        return tuple(torch.stack(batch).to(DEVICE) for batch in batches)  # type:ignore

    __len__ = lambda self: len(self.deque)

