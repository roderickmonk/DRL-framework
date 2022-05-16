import logging

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter


class LocallyConnected1d(torch.nn.Module):
    """"""

    @staticmethod
    def out_size(in_size: int, kernel_size: int, stride: int) -> int:
        return (in_size - kernel_size) // stride + 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        in_size: int,
        kernel_size: int,
        stride: int,
        bias=False,
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride

        out_size = self.out_size(in_size, kernel_size, stride)

        self.weight = Parameter(
            torch.randn(1, out_channels, in_channels, out_size, kernel_size)
        )

        if bias:
            self.bias = Parameter(torch.randn(1, out_channels, out_size))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: Tensor) -> Tensor:

        # region
        _log = logging.debug

        _log(f"x.0: {x}")
        _log(f"x.shape.0: {x.shape}")
        # endregion

        x = x.unfold(2, self.kernel_size, self.stride).unsqueeze(4)

        # region
        _log(f"x.1: {x}")
        _log(f"x.shape.1: {x.shape}")
        # endregion

        x = x.contiguous().view(*x.size()[:-2], -1)

        # region
        _log(f"x.2: {x}")
        _log(f"x.shape.2: {x.shape}")
        # endregion

        out = (x.unsqueeze(1) * self.weight).sum([2, -1])

        return out if self.bias is None else out + self.bias


class LocallyConnected1d_ML(torch.nn.Module):
    def __init__(
        self,
        *,
        channels: list[int],
        in_size: int,
        kernel_size: int,
        stride: int,
        activation=F.relu,
    ):

        super().__init__()

        self.channels = channels
        self.in_size = in_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.activation = activation

        self.input_sizes = self._input_sizes()

        self.layers = torch.nn.ModuleList(
            [
                LocallyConnected1d(
                    channel,
                    channels[i + 1],
                    in_size=self.input_sizes[i],
                    kernel_size=kernel_size,
                    stride=stride,
                )
                for i, channel in enumerate(channels[0:-1])
            ]
        )

    def last_output_size(self):
        return self.input_sizes[-1]

    def _input_sizes(self):

        sizes = [self.in_size]
        for i in range(1, len(self.channels)):
            next_input_size = (sizes[-1] - self.kernel_size) // (self.stride) + 1
            sizes.append(next_input_size)
        return sizes

    def forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
        return x
