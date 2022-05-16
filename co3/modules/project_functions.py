import torch
from constants import DEVICE
from torch import nn
from typing import Any, List
from itertools import count

float_tensor = lambda x: torch.tensor(x, dtype=torch.float32, device=DEVICE)


def initialize_weights(m):

    if isinstance(m, nn.Conv1d):
        nn.init.kaiming_uniform_(m.weight.data, nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)


def repeat_last(l: List):

    ll = l

    i = count(0)
    l_len = len(l)

    while True:

        if (j := next(i)) < l_len:
            yield ll[j]
        else:
            yield ll[-1]
