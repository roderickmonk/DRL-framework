from collections import OrderedDict

from networks.fully_connected import FullyConnected
from torch import nn as nn


class DqnNetwork(nn.Module):
    """"""

    def __init__(
        self,
        *,
        config,
        state_size,
        action_size,
    ):

        super().__init__()

        model_config = config.agent.torch_models.main

        self.model = FullyConnected(
            input_size=state_size,
            hidden_dims=model_config.hidden_dims,
            output_size=action_size,
            activations=model_config.activations,
        )

    def forward(self, state):
        return self.model(state)
