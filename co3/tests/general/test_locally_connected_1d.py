import logging
from itertools import count

import numpy as np
import pytest
import torch
from env_modules.evaluate_env import EvaluateEnv
from networks.locally_connected_1d import LocallyConnected1d, LocallyConnected1d_ML
from time_breaks_0 import TIME_BREAKS
from torch import Tensor

logging.basicConfig(
    format="[%(levelname)-5s] %(message)s", level=logging.INFO, datefmt=""
)

torch.set_printoptions(
    precision=4,
    threshold=None,
    edgeitems=4,
    linewidth=100,
    profile=None,
    sci_mode=False,
)


def fixed_weights(
    out_channels: int, in_channels: int, out_size: int, kernel_size: int
) -> torch.nn.Parameter:  # type:ignore

    weights = torch.empty(1, out_channels, in_channels, out_size, kernel_size)

    fill_data = count(5, 5)
    for i in range(out_channels):
        weights[0][i] = torch.full(
            (1, in_channels, out_size, kernel_size), next(fill_data)
        )

    print("\n***   weights   ***")
    print(weights)
    print(f"{weights.shape=}")

    return torch.nn.Parameter(weights)  # type:ignore


def lcn(
    *, in_channels: int, out_channels: int, in_size: int, kernel_size, stride: int
) -> LocallyConnected1d:

    lcn = LocallyConnected1d(
        in_channels,
        out_channels,
        in_size=in_size,
        kernel_size=kernel_size,
        stride=stride,
    )
    out_size = lcn.out_size(in_size, kernel_size, stride)

    lcn.weight = fixed_weights(out_channels, in_channels, out_size, kernel_size)

    return lcn


def test_LocallyConnected1d_0():

    batch_size = 1
    in_channels = 1
    out_channels = 1
    in_size = 10
    kernel_size = 1
    stride = 1

    x = torch.full((batch_size, in_channels, in_size), 1)

    out = lcn(
        in_channels=in_channels,
        out_channels=out_channels,
        in_size=in_size,
        kernel_size=kernel_size,
        stride=stride,
    )(x)

    expected = Tensor([[[5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]]])

    print(out)
    print(out.shape)

    assert out.shape == expected.shape
    assert torch.equal(out, expected)


def test_LocallyConnected1d_1():

    batch_size = 1
    in_channels = 1
    out_channels = 1
    in_size = 10
    kernel_size = 2
    stride = 1

    x = torch.full((batch_size, in_channels, in_size), 3)
    print("\nx: \n", x)

    out = lcn(
        in_channels=in_channels,
        out_channels=out_channels,
        in_size=in_size,
        kernel_size=kernel_size,
        stride=stride,
    )(x)

    expected = Tensor([[[30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0]]])

    print(out)
    print(out.shape)

    assert out.shape == expected.shape
    assert torch.equal(out, expected)


def test_LocallyConnected1d_2():

    batch_size = 1
    in_channels = 1
    out_channels = 1
    in_size = 10
    kernel_size = 3
    stride = 1

    x = torch.full((1, in_channels, in_size), 2)
    print("\nx: \n", x)

    out = lcn(
        in_channels=in_channels,
        out_channels=out_channels,
        in_size=in_size,
        kernel_size=kernel_size,
        stride=stride,
    )(x)

    expected = torch.Tensor([[[30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0]]])

    print(out)
    print(out.shape)

    assert out.shape == expected.shape
    assert torch.equal(out, expected)


def test_LocallyConnected1d_3():

    batch_size = 1
    in_channels = 1
    out_channels = 2
    in_size = 10
    kernel_size = 1
    stride = 1

    x = torch.full((batch_size, in_channels, in_size), 20)
    print("\n***   test input   ***")
    print(x)

    out = lcn(
        in_channels=in_channels,
        out_channels=out_channels,
        in_size=in_size,
        kernel_size=kernel_size,
        stride=stride,
    )(x)

    expected = torch.Tensor(
        [
            [
                [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
                [200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0],
            ]
        ]
    )

    print("\n***   test output   ***")
    print(out)
    print(out.shape)

    assert out.shape == expected.shape
    assert torch.equal(out, expected)


def test_LocallyConnected1d_4():

    batch_size = 1
    in_channels = 1
    out_channels = 10
    in_size = 10
    kernel_size = 1
    stride = 1

    x = torch.full((batch_size, in_channels, in_size), 2)
    print("\n***   test input   ***")
    print(x)

    out: Tensor = lcn(
        in_channels=in_channels,
        out_channels=out_channels,
        in_size=in_size,
        kernel_size=kernel_size,
        stride=stride,
    )(x)

    expected = Tensor(
        [
            [
                [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
                [20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0],
                [30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0],
                [40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0],
                [50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0],
                [60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0],
                [70.0, 70.0, 70.0, 70.0, 70.0, 70.0, 70.0, 70.0, 70.0, 70.0],
                [80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0],
                [90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0],
                [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            ]
        ]
    )

    print("\n***   test output   ***")
    print(out)
    print(out.shape)

    assert torch.equal(out, expected)
    assert out.shape == expected.shape


def test_LocallyConnected1d_5():

    batch_size = 64
    in_channels = 1
    out_channels = 10
    in_size = 10
    kernel_size = 1
    stride = 1

    x = torch.full((batch_size, in_channels, in_size), 100)

    out: Tensor = lcn(
        in_channels=in_channels,
        out_channels=out_channels,
        in_size=in_size,
        kernel_size=kernel_size,
        stride=stride,
    )(x)

    print(out.shape)

    assert list(out.shape) == [64, 10, 10]


def test_LocallyConnected1d_6():

    batch_size = 64
    in_channels = 1
    out_channels = 20
    in_size = 10
    kernel_size = 1
    stride = 1

    x = torch.full((batch_size, in_channels, in_size), 100)

    out: Tensor = lcn(
        in_channels=in_channels,
        out_channels=out_channels,
        in_size=in_size,
        kernel_size=kernel_size,
        stride=stride,
    )(x)

    print(out.shape)

    assert list(out.shape) == [64, 20, 10]


def test_LocallyConnected1d_7():

    batch_size = 64
    in_channels = 1
    out_channels = 20
    in_size = 20
    kernel_size = 1
    stride = 1

    x = torch.full((batch_size, in_channels, in_size), 100)

    out: Tensor = lcn(
        in_channels=in_channels,
        out_channels=out_channels,
        in_size=in_size,
        kernel_size=kernel_size,
        stride=stride,
    )(x)

    print(out.shape)

    assert list(out.shape) == [64, 20, 20]


def test_LocallyConnected1d_8():

    batch_size = 64
    in_channels = 1
    out_channels = 20
    in_size = 400
    kernel_size = 1
    stride = 1

    x = torch.full((batch_size, in_channels, in_size), 100)

    out: Tensor = lcn(
        in_channels=in_channels,
        out_channels=out_channels,
        in_size=in_size,
        kernel_size=kernel_size,
        stride=stride,
    )(x)

    print(out.shape)

    assert list(out.shape) == [64, 20, 400]


def test_LocallyConnected1d_9():

    batch_size = 64
    in_channels = 1
    out_channels = 20
    in_size = 100
    kernel_size = 2
    stride = 1

    x = torch.full((batch_size, in_channels, in_size), 100)

    out: Tensor = lcn(
        in_channels=in_channels,
        out_channels=out_channels,
        in_size=in_size,
        kernel_size=kernel_size,
        stride=stride,
    )(x)

    print(out.shape)

    assert list(out.shape) == [64, 20, 99]


def test_LocallyConnected1d_10():

    batch_size = 64
    in_channels = 1
    out_channels = 20
    in_size = 100
    kernel_size = 4
    stride = 1

    x = torch.full((batch_size, in_channels, in_size), 100)

    out: Tensor = lcn(
        in_channels=in_channels,
        out_channels=out_channels,
        in_size=in_size,
        kernel_size=kernel_size,
        stride=stride,
    )(x)

    print(out.shape)

    assert list(out.shape) == [64, 20, 97]


def test_LocallyConnected1d_11():

    batch_size = 64
    in_channels = 1
    out_channels = 20
    in_size = 100
    kernel_size = 40
    stride = 1

    x = torch.full((batch_size, in_channels, in_size), 100)

    out: Tensor = lcn(
        in_channels=in_channels,
        out_channels=out_channels,
        in_size=in_size,
        kernel_size=kernel_size,
        stride=stride,
    )(x)

    print(out.shape)

    assert list(out.shape) == [64, 20, 61]


def test_LocallyConnected1d_12():

    batch_size = 64
    in_channels = 1
    out_channels = 20
    in_size = 100
    kernel_size = 1
    stride = 2

    x = torch.full((batch_size, in_channels, in_size), 100)

    out: Tensor = lcn(
        in_channels=in_channels,
        out_channels=out_channels,
        in_size=in_size,
        kernel_size=kernel_size,
        stride=stride,
    )(x)

    print(out.shape)

    assert list(out.shape) == [64, 20, 50]


def test_LocallyConnected1d_13():

    batch_size = 64
    in_channels = 1
    out_channels = 20
    in_size = 101
    kernel_size = 1
    stride = 2

    x = torch.full((batch_size, in_channels, in_size), 100)

    out: Tensor = lcn(
        in_channels=in_channels,
        out_channels=out_channels,
        in_size=in_size,
        kernel_size=kernel_size,
        stride=stride,
    )(x)

    print(out.shape)

    assert list(out.shape) == [64, 20, 51]


def test_LocallyConnected1d_14():

    batch_size = 64
    in_channels = 1
    out_channels = 20
    in_size = 99
    kernel_size = 1
    stride = 2

    x = torch.full((batch_size, in_channels, in_size), 100)

    out: Tensor = lcn(
        in_channels=in_channels,
        out_channels=out_channels,
        in_size=in_size,
        kernel_size=kernel_size,
        stride=stride,
    )(x)

    print(out.shape)

    assert list(out.shape) == [64, 20, 50]


def test_LocallyConnected1d_15():

    batch_size = 64
    in_channels = 1
    out_channels = 20
    in_size = 100
    kernel_size = 1
    stride = 3

    x = torch.full((batch_size, in_channels, in_size), 100)

    out: Tensor = lcn(
        in_channels=in_channels,
        out_channels=out_channels,
        in_size=in_size,
        kernel_size=kernel_size,
        stride=stride,
    )(x)

    print(out.shape)

    assert list(out.shape) == [64, 20, 34]


def test_LocallyConnected1d_16():

    batch_size = 64
    in_channels = 1
    out_channels = 20
    in_size = 101
    kernel_size = 1
    stride = 3

    x = torch.full((batch_size, in_channels, in_size), 100)

    out: Tensor = lcn(
        in_channels=in_channels,
        out_channels=out_channels,
        in_size=in_size,
        kernel_size=kernel_size,
        stride=stride,
    )(x)

    print(out.shape)

    assert list(out.shape) == [64, 20, 34]


def test_LocallyConnected1d_17():

    batch_size = 64
    in_channels = 1
    out_channels = 20
    in_size = 102
    kernel_size = 1
    stride = 3

    x = torch.full((batch_size, in_channels, in_size), 100)

    out: Tensor = lcn(
        in_channels=in_channels,
        out_channels=out_channels,
        in_size=in_size,
        kernel_size=kernel_size,
        stride=stride,
    )(x)

    print(out.shape)

    assert list(out.shape) == [64, 20, 34]


def test_LocallyConnected1d_18():

    batch_size = 64
    in_channels = 1
    out_channels = 20
    in_size = 103
    kernel_size = 1
    stride = 3

    x = torch.full((batch_size, in_channels, in_size), 100)

    out: Tensor = lcn(
        in_channels=in_channels,
        out_channels=out_channels,
        in_size=in_size,
        kernel_size=kernel_size,
        stride=stride,
    )(x)

    print(out.shape)

    assert list(out.shape) == [64, 20, 35]


def test_LocallyConnected1d_19():

    batch_size = 64
    in_channels = 1
    out_channels = 20
    in_size = 99
    kernel_size = 1
    stride = 3

    x = torch.full((batch_size, in_channels, in_size), 100)

    out: Tensor = lcn(
        in_channels=in_channels,
        out_channels=out_channels,
        in_size=in_size,
        kernel_size=kernel_size,
        stride=stride,
    )(x)

    print(out.shape)

    assert list(out.shape) == [64, 20, 33]


def test_LocallyConnected1d_20():

    batch_size = 64
    in_channels = 2
    out_channels = 20
    in_size = 100
    kernel_size = 1
    stride = 1

    x = torch.full((batch_size, in_channels, in_size), 100)

    out: Tensor = lcn(
        in_channels=in_channels,
        out_channels=out_channels,
        in_size=in_size,
        kernel_size=kernel_size,
        stride=stride,
    )(x)

    print(out.shape)

    assert list(out.shape) == [64, 20, 100]


def test_LocallyConnected1d_21():

    batch_size = 64
    in_channels = 15
    out_channels = 20
    in_size = 100
    kernel_size = 1
    stride = 1

    x = torch.full((batch_size, in_channels, in_size), 100)

    out: Tensor = lcn(
        in_channels=in_channels,
        out_channels=out_channels,
        in_size=in_size,
        kernel_size=kernel_size,
        stride=stride,
    )(x)

    print(out.shape)

    assert list(out.shape) == [64, 20, 100]


def test_LocallyConnected1d_22():

    batch_size = 64
    in_channels = 5
    out_channels = 20
    in_size = 100
    kernel_size = 1
    stride = 2

    x = torch.full((batch_size, in_channels, in_size), 100)

    out: Tensor = lcn(
        in_channels=in_channels,
        out_channels=out_channels,
        in_size=in_size,
        kernel_size=kernel_size,
        stride=stride,
    )(x)

    print(out.shape)

    assert list(out.shape) == [64, 20, 50]


def test_LocallyConnected1d_23():

    batch_size = 64
    in_channels = 5
    out_channels = 20
    in_size = 100
    kernel_size = 1
    stride = 3

    x = torch.full((batch_size, in_channels, in_size), 100)

    out: Tensor = lcn(
        in_channels=in_channels,
        out_channels=out_channels,
        in_size=in_size,
        kernel_size=kernel_size,
        stride=stride,
    )(x)

    print(out.shape)

    assert list(out.shape) == [64, 20, 34]


def test_LocallyConnected1d_24():

    batch_size = 64
    in_channels = 5
    out_channels = 20
    in_size = 100
    kernel_size = 2
    stride = 3

    x = torch.full((batch_size, in_channels, in_size), 100)

    out: Tensor = lcn(
        in_channels=in_channels,
        out_channels=out_channels,
        in_size=in_size,
        kernel_size=kernel_size,
        stride=stride,
    )(x)

    print(out.shape)

    assert list(out.shape) == [64, 20, 33]


def test_LocallyConnected1d_25():

    batch_size = 64
    in_channels = 5
    out_channels = 20
    in_size = 100
    kernel_size = 3
    stride = 3

    x = torch.full((batch_size, in_channels, in_size), 100)

    out: Tensor = lcn(
        in_channels=in_channels,
        out_channels=out_channels,
        in_size=in_size,
        kernel_size=kernel_size,
        stride=stride,
    )(x)

    print(out.shape)

    assert list(out.shape) == [64, 20, 33]


def test_LocallyConnected1d_26():

    batch_size = 64
    in_channels = 5
    out_channels = 20
    in_size = 100
    kernel_size = 4
    stride = 3

    x = torch.full((batch_size, in_channels, in_size), 100)

    out: Tensor = lcn(
        in_channels=in_channels,
        out_channels=out_channels,
        in_size=in_size,
        kernel_size=kernel_size,
        stride=stride,
    )(x)

    print(out.shape)

    assert list(out.shape) == [64, 20, 33]


def test_LocallyConnected1d_27():

    batch_size = 64
    in_channels = 5
    out_channels = 20
    in_size = 100
    kernel_size = 5
    stride = 3

    x = torch.full((batch_size, in_channels, in_size), 100)

    out: Tensor = lcn(
        in_channels=in_channels,
        out_channels=out_channels,
        in_size=in_size,
        kernel_size=kernel_size,
        stride=stride,
    )(x)

    print(out.shape)

    assert list(out.shape) == [64, 20, 32]


def test_LocallyConnected1d_28():

    batch_size = 64
    in_channels = 5
    out_channels = 20
    in_size = 100
    kernel_size = 10
    stride = 3

    x = torch.full((batch_size, in_channels, in_size), 100)

    out: Tensor = lcn(
        in_channels=in_channels,
        out_channels=out_channels,
        in_size=in_size,
        kernel_size=kernel_size,
        stride=stride,
    )(x)

    print(out.shape)

    assert list(out.shape) == [64, 20, 31]


def test_LocallyConnected1d_29():

    batch_size = 64
    in_channels = 5
    out_channels = 20
    in_size = 99
    kernel_size = 10
    stride = 3

    x = torch.full((batch_size, in_channels, in_size), 100)

    out: Tensor = lcn(
        in_channels=in_channels,
        out_channels=out_channels,
        in_size=in_size,
        kernel_size=kernel_size,
        stride=stride,
    )(x)

    print(out.shape)

    assert list(out.shape) == [64, 20, 30]


def test_LocallyConnected1d_30():

    batch_size = 1
    in_channels = 2
    out_channels = 10
    in_size = 10
    kernel_size = 1
    stride = 1

    x = torch.full((batch_size, in_channels, in_size), 2)

    out: Tensor = lcn(
        in_channels=in_channels,
        out_channels=out_channels,
        in_size=in_size,
        kernel_size=kernel_size,
        stride=stride,
    )(x)

    expected = Tensor(
        [
            [
                [20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0],
                [40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0],
                [60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0],
                [80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0],
                [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
                [120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0],
                [140.0, 140.0, 140.0, 140.0, 140.0, 140.0, 140.0, 140.0, 140.0, 140.0],
                [160.0, 160.0, 160.0, 160.0, 160.0, 160.0, 160.0, 160.0, 160.0, 160.0],
                [180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0],
                [200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0],
            ]
        ]
    )

    print(out)
    print(out.shape)

    assert torch.equal(out, expected)
    assert out.shape == expected.shape


def test_LocallyConnected1d_31():

    batch_size = 1
    in_channels = 3
    out_channels = 10
    in_size = 10
    kernel_size = 1
    stride = 1

    x = torch.full((batch_size, in_channels, in_size), 2)

    out: Tensor = lcn(
        in_channels=in_channels,
        out_channels=out_channels,
        in_size=in_size,
        kernel_size=kernel_size,
        stride=stride,
    )(x)

    expected = Tensor(
        [
            [
                [30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0],
                [60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0],
                [90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0],
                [120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0],
                [150.0, 150.0, 150.0, 150.0, 150.0, 150.0, 150.0, 150.0, 150.0, 150.0],
                [180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0],
                [210.0, 210.0, 210.0, 210.0, 210.0, 210.0, 210.0, 210.0, 210.0, 210.0],
                [240.0, 240.0, 240.0, 240.0, 240.0, 240.0, 240.0, 240.0, 240.0, 240.0],
                [270.0, 270.0, 270.0, 270.0, 270.0, 270.0, 270.0, 270.0, 270.0, 270.0],
                [300.0, 300.0, 300.0, 300.0, 300.0, 300.0, 300.0, 300.0, 300.0, 300.0],
            ]
        ]
    )

    print(out)
    print(out.shape)

    assert torch.equal(out, expected)
    assert out.shape == expected.shape
