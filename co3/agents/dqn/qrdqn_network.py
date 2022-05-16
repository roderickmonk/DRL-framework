import importlib
import itertools as it
import random
from collections import OrderedDict

import torch
from networks.fully_connected import FullyConnected
from torch import nn as nn


class QrdqnNetwork(nn.Module):
    """"""

    def __init__(self, *, state_size, action_size, config):

        super().__init__()

        self.quantile_resolution = config.agent.quantile_resolution
        self.action_size = action_size

        model_config = config.agent.torch_models.main

        self.model = FullyConnected(
            input_size=state_size,
            hidden_dims=model_config.hidden_dims,
            output_size=action_size * self.quantile_resolution,
            activations=model_config.activations,
        )

        self.trace_active = False
        if config.nn_trace:

            self.nn_trace = config.nn_trace

            if bool(self.nn_trace) and hasattr(self.nn_trace, "network_trace"):
                self.trace_active = True
                self.max_trace = self.nn_trace.count
                self._trace_counter = it.count(0)

    def forward(self, state, episode=None):
        """"""

        def trace():

            if (
                self.trace_active
                and not episode is None
                and next(self._trace_counter) < self.max_trace
            ):

                data = [
                    episode,
                    x.detach().cpu().squeeze().numpy(),
                    v.detach().cpu().squeeze().numpy(),
                ]

                self.nn_trace.network_trace.write(data)

        x = self.model(state)

        v = x.view(-1, self.action_size, self.quantile_resolution)

        trace()

        return v

    select_action = lambda self, *, state, training, eps: (
        int(
            self.forward(state).mean(2).max(1)[1]
            if not training or random.random() > eps
            else torch.randint(0, 2, (1,))
        )
    )
