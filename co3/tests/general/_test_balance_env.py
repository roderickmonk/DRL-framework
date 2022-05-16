import logging

from sentient_util import cfg
import numpy as np
import pytest
from balance_env import BalanceEnv
import pytest

logging.basicConfig(
    format="[%(levelname)-5s] %(message)s", level=logging.INFO, datefmt=""
)

np.set_printoptions(precision=14, floatmode="fixed", edgeitems=25)

nominal_config = cfg.ConfigObj(
    {
        "episode_length": 10,
        "order_depths": np.array([0.0, 0.05, 0.10, 0.15, 0.20]),
        "il": 0.2,
        "ql": 0.2,
        "reward_offset": 0.0,
    }
)

pytest.skip("skipping BalanceEnv tests", allow_module_level=True)


class __TestEnv(BalanceEnv):
    """ BalanceEnv is abstract, hence this dummy class """

    def __init__(self, *, config, start_step):
        super().__init__(config=config, start_step=start_step)

        self._episode_step = it.count(1)

    def reset(self):
        pass

    get_it = lambda self: {"bid": 0.25, "ask": 0.25}
    get_state = lambda self: []

    def reset(self):
        pass

    def generator(self):
        pass

    def _tick(self):

        while True:
            self.episode_step = next(self._episode_step)
            yield


# An empty object
config = type("test", (object,), {})()


@pytest.fixture
def set_nominal_config():
    """ Restore config to the nominal """
    config.__dict__.update(nominal_config.__dict__)


def test_balance_env_0(set_nominal_config):

    env = __TestEnv(config=config, start_step=1)

    assert (0.2, 0.0, 0.2) == env.get_reward(
        it_bid=0.25,
        it_ask=0.25,
        bid_depth=0.0,
        ask_depth=0.0,
        bid_pf=1.0,
        ask_pf=1.0,
        balance=10.0,
        ql=config.ql,
        il=config.il,
    )


def test_balance_env_1(set_nominal_config):

    env = __TestEnv(config=config, start_step=1)

    assert (0.2, 0.0, 0.2) == env.get_reward(
        it_bid=0.25,
        it_ask=0.25,
        bid_depth=0.0,
        ask_depth=0.0,
        bid_pf=1.0,
        ask_pf=1.0,
        balance=10.0,
        ql=config.ql,
        il=config.il,
    )


def test_balance_env_2(set_nominal_config):

    env = __TestEnv(config=config, start_step=1)

    result = [
        reward_datum
        for action in range(len(config.order_depths))
        if (
            reward_datum := env.get_reward(
                it_bid=0.25,
                it_ask=0.25,
                bid_depth=0.0,
                ask_depth=0.0,
                bid_pf=1.0,
                ask_pf=1.0,
                balance=10.0,
                ql=config.ql,
                il=config.il,
            )
        )
        is not None
    ]

    print(f"{result=}")

    expected_result = [
        (0.2, 0.0, 0.2),
        (0.2, 0.0, 0.2),
        (0.2, 0.0, 0.2),
        (0.2, 0.0, 0.2),
        (0.2, 0.0, 0.2),
    ]
    assert expected_result == result


def test_balance_env_3(set_nominal_config):

    env = __TestEnv(config=config, start_step=1)

    result = [
        reward_datum
        for action in range(len(config.order_depths))
        if (
            reward_datum := env.get_reward(
                it_bid=0.25,
                it_ask=0.25,
                bid_depth=0.0,
                ask_depth=0.0,
                bid_pf=1.0,
                ask_pf=1.0,
                balance=10.0,
                ql=config.ql,
                il=config.il,
            )
        )
        is not None
    ]

    print(f"{result=}")

    expected_result = [
        (0.2, 0.0, 0.2),
        (0.2, 0.0, 0.2),
        (0.2, 0.0, 0.2),
        (0.2, 0.0, 0.2),
        (0.2, 0.0, 0.2),
    ]
    assert expected_result == result


def test_balance_env_4(set_nominal_config):

    bid_pf = 1 / (config.order_depths + np.argmax(config.order_depths))
    ask_pf = 1 / (config.order_depths + np.argmax(config.order_depths))

    env = __TestEnv(config=config, start_step=1)

    result = []

    for i in range(config.episode_length):

        next(env._tick())

        result.append(
            env.step(bid_action=0, ask_action=0, bid_pf=bid_pf, ask_pf=ask_pf)
        )

    expected_result = [
        (0.05, 0.2, 0.0, 0.2, 1, False),
        (0.05, 0.0, 0.2, 0.0, 2, False),
        (0.05, 0.2, 0.0, 0.2, 3, False),
        (0.05, 0.0, 0.2, 0.0, 4, False),
        (0.05, 0.2, 0.0, 0.2, 5, False),
        (0.05, 0.0, 0.2, 0.0, 6, False),
        (0.05, 0.2, 0.0, 0.2, 7, False),
        (0.05, 0.0, 0.2, 0.0, 8, False),
        (0.05, 0.2, 0.0, 0.2, 9, False),
        (0.05, 0.0, 0.2, 0.0, 10, True),
    ]

    assert expected_result == result

