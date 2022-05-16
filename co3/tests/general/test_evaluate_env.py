import logging

import numpy as np
import pytest
from env_modules.evaluate_env import EvaluateEnv
from time_breaks_0 import TIME_BREAKS

logging.basicConfig(
    format="[%(levelname)-5s] %(message)s", level=logging.INFO, datefmt=""
)

np.set_printoptions(precision=14, floatmode="fixed", edgeitems=25) #type:ignore


def test_evaluate_env_0():

    assert (0.0, 0.2, 1.0, 0.0) == EvaluateEnv.get_reward(
        it=0.25,
        action=0,
        pf=np.array([1.0]),
        order_depths=np.array([0.0]),
        ql=0.2,
        reward_offset=0,
    )


def test_evaluate_env_1():

    order_depths = np.array([0.1])
    pf = 1 / (order_depths + np.argmax(order_depths))

    assert (0.0, 0.15, 10.0, 0.1) == EvaluateEnv.get_reward(
        it=0.25, action=0, pf=pf, order_depths=order_depths, ql=0.2, reward_offset=0,
    )


def test_evaluate_env_2():

    order_depths = np.array([0.0, 0.05, 0.10, 0.15, 0.20])
    pf = 1 / (order_depths + np.argmax(order_depths))

    result = [
        EvaluateEnv.get_reward(
            it=0.25,
            order_depths=order_depths,
            action=action,
            pf=pf,
            reward_offset=0,
            ql=0.2,
        )
        for action in range(len(order_depths))
    ]

    expected_result = [
        (0.0, 0.2, 0.25, 0.0),
        (-0.0006172839506172825, 0.2, 0.2469135802469136, 0.05),
        (-0.013414634146341461, 0.15, 0.24390243902439027, 0.1),
        (-0.02590361445783133, 0.1, 0.24096385542168672, 0.15),
        (-0.0380952380952381, 0.04999999999999999, 0.23809523809523808, 0.2),
    ]

    assert expected_result == result


def test_evaluate_env_3():

    order_depths = np.array([0.0, 0.05, 0.10, 0.15, 0.20])
    pf = 1 / (order_depths + np.argmax(order_depths))
    ql = 0.2
    reward_offset = 0.0

    result = EvaluateEnv._step(
        action=0,
        pf=pf,
        it=1.0,
        episode_step=1,
        order_depths=order_depths,
        next_state=None,
        ql=ql,
        reward_offset=reward_offset,
    )

    _, reward, done, info = result

    assert reward == 0.0
    assert info["fill_size"] == 0.2
    assert info["pf"] == 0.25
    assert info["episode_step"] == 1

