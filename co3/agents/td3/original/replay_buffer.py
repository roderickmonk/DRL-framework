from collections import deque, namedtuple

import numpy as np
import torch
from constants import DEVICE, Experience, FloatTensor


class ReplayBuffer(object):
    def __init__(self, state_dim, action_size, batch_size, buffer_size):

        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((buffer_size, state_dim))
        self.action = np.zeros((buffer_size, action_size))
        self.next_state = np.zeros((buffer_size, state_dim))
        self.reward = np.zeros((buffer_size, 1))
        self.done = np.zeros((buffer_size, 1))

        self.deque = deque(maxlen=buffer_size)
        self.rng = np.random.default_rng()

    def __add__(self, experience: Experience):

        self.deque.append(experience)

        state, action, next_state, reward, done = experience

        self.state[self.ptr] = state.cpu()
        self.action[self.ptr] = action.cpu()
        self.next_state[self.ptr] = next_state.cpu()
        self.reward[self.ptr] = reward.cpu()
        self.done[self.ptr] = done.cpu()

        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self):

        indices = self.rng.integers(low=0, high=self.size, size=self.batch_size)

        # ind = np.random.Generator.integers(0, self.size, size=self.batch_size)
        # ind = np.random.randint(0, self.size, size=self.batch_size)

        batches = zip(*[self.deque[idx] for idx in indices])
        return [torch.stack(batch).to(DEVICE) for batch in batches]

        return (
            self.state[indices],
            self.action[indices],
            self.next_state[indices],
            self.reward[indices],
            self.done[indices],
        )

    __len__ = lambda self: len(self.deque) if hasattr(self, "deque") else self.size

