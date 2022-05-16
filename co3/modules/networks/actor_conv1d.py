import logging
from dataclasses import dataclass
from typing import Any, Union

import torch
from constants import DEVICE
from networks.fully_connected import FullyConnected
from pydantic_config import DDPG_SimEnv1_Config, TD3_SimEnv1_Config

_log = logging.critical

torch.set_printoptions(precision=10, sci_mode=True)
from networks.conv1d_network import conv1d_network
from devtools import debug

ProcessConfigType = Union[DDPG_SimEnv1_Config, TD3_SimEnv1_Config]


@dataclass(eq=False, order=False)
class Actor(torch.nn.Module):
    env: Any
    config: ProcessConfigType
    output_size_explicit: bool = True

    def __post_init__(self):

        super().__init__()

        self.torch_model = self.config.agent.torch_models.Actor

        vector_length = (
            self.env.env.observation_space.shape[1] - 1
        )  # -1 drops the non-vector data

        self.buy_model, _ = conv1d_network(
            filters=self.torch_model.conv_filters,
            kernel_size=self.torch_model.kernel_size,
            vector_length=vector_length,
        )
        self.sell_model, self.conv_out_size = conv1d_network(
            filters=self.torch_model.conv_filters,
            kernel_size=self.torch_model.kernel_size,
            vector_length=vector_length,
        )

        self.fc_model = FullyConnected(
            input_size=2 * self.conv_out_size + 2,  # +2 adds in the extra fields
            hidden_dims=list(self.torch_model.hidden_dims),
            output_size=self.env.action_space.shape[0]
            if self.output_size_explicit
            else None,
            activations=list(self.torch_model.activations),
        )

        self.to(DEVICE)

    def forward(self, x):

        # debug(x)
        # debug(self)
        # breakpoint()

        return torch.tanh(
            self.fc_model(
                torch.concat(
                    (
                        # Flatten the output from both convolutions
                        self.buy_model(x[:, 0, :-1].unsqueeze(1)).view(
                            -1, self.conv_out_size
                        ),
                        self.sell_model(x[:, 1, :-1].unsqueeze(1)).view(
                            -1, self.conv_out_size
                        ),
                        # Tack on the funds delta and the balance
                        x[:, 0, -1].unsqueeze(1),
                        x[:, 1, -1].unsqueeze(1),
                    ),
                    1,
                )
            )
        )
