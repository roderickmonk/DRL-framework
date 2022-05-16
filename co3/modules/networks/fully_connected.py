import itertools as it
from typing import List, Optional

from constants import DEVICE
from project_functions import repeat_last
from sentient_util.exceptions import InvalidConfiguration
from torch import nn as nn
from torch.nn import Sequential


def FullyConnected(
    *,
    input_size: int,
    hidden_dims: List[int],
    activations: List[str],
    output_size: Optional[int] = None,
):

    if len(hidden_dims) == 0:
        raise InvalidConfiguration(f"At least one hidden layer is required")

    activation = repeat_last([eval(activation) for activation in activations])    

    layer_sizes = [input_size] + list(hidden_dims)

    layers = []

    if output_size:

        for i in range(len(layer_sizes) - 1):
            layers.extend(
                [
                    nn.Linear(layer_sizes[i], layer_sizes[i + 1]),
                    nn.LayerNorm(layer_sizes[i + 1]),
                    next(activation)(),
                ]
            )

        layers.extend([nn.Linear(layer_sizes[-1], output_size)])

    else:

        for i in range(len(layer_sizes) - 2):
            layers.extend(
                [
                    nn.Linear(layer_sizes[i], layer_sizes[i + 1]),
                    nn.LayerNorm(layer_sizes[i + 1]),
                    next(activation)(),
                ]
            )
        else:
            layers.extend(
                [
                    nn.Linear(layer_sizes[-2], layer_sizes[-1]),
                    nn.LayerNorm(layer_sizes[-1]),
                ]
            )

    return Sequential(*layers).to(DEVICE)

