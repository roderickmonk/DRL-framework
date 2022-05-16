import math
from typing import Tuple, Union

import torch
import torch.nn as nn
from omegaconf import ListConfig


def conv1d_network(
    *,
    filters: Union[ListConfig, list[int]],
    kernel_size: int,
    vector_length: int,
) -> Tuple[nn.Sequential, int]:

    layers = [nn.Conv1d(1, filters[0], kernel_size), nn.ReLU()]

    for i in range(len(filters) - 1):
        layers += [nn.Conv1d(filters[i], filters[i + 1], kernel_size), nn.ReLU()]

    net = nn.Sequential(*layers)

    # Incidently record the size of the network's output
    with torch.no_grad():
        conv_out_size = math.prod(
            net(torch.randn(1, 1, vector_length)).squeeze(0).shape
        )

    return net, conv_out_size
