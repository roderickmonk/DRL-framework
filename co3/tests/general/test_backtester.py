import logging
import sys
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
from itertools import islice
from typing import Any
import pytest

import numpy as np
from devtools import debug
from env_modules.backtester import BackTester
from sentient_util.pydantic_config import SimEnv0_Config

episodes = 10
episode_length = 10000

env_config: dict[str, Any] = {
    "QL": 0,
    "IL": 0,
    "orderbook_entries": 0,
    "query": {
        "envId": 0,
        "exchange": "bittrex",
        "market": "btc-eth",
        "start_range": "2021-10-01T00:00:00",
        "end_range": "2021-10-15T00:00:00",
    },
    "action_space_low": [-9, -9],
    "action_space_high": [-6, -6],
    "episode_type": "fixed_length",  # variable_length
    "episode_length": episode_length,
    "episodes": episodes,
    "front_load": False,
    "orderbooks_with_trades_collection": "V3-recovered-orderbooks-with-trades-btc-eth",
}

@pytest.mark.skip
def test_backtester_load():

    episode_length = 2000
    episodes = 5
    total_test_orderbooks = episode_length * episodes
    env_override: dict[str, Any] = {
        "episode_length": episode_length,
        "episodes": episodes,
        "front_load": True,
    }

    _, testing_orderbooks = BackTester.load(
        env_config=SimEnv0_Config(**env_config | env_override)
    )

    loops = 20

    batch = []
    for _ in range(loops):
        batch.append(
            [x.id for x in list(islice(testing_orderbooks, 0, total_test_orderbooks))]
        )

    # Ensure that all batches contain the same thing
    assert all(element == batch[0] for element in batch[1:])

    # Skew the islice to NOT repeat on the 10000 boundary
    _, testing_orderbooks = BackTester.load(
        env_config=SimEnv0_Config(**env_config | env_override)
    )
    batch = []
    for _ in range(loops):
        batch.append(
            [
                x.id
                for x in list(islice(testing_orderbooks, 0, total_test_orderbooks - 1))
            ]
        )
    # Must not get equality this time
    assert not all(element == batch[0] for element in batch[1:])

    # A second such test
    _, testing_orderbooks = BackTester.load(
        env_config=SimEnv0_Config(**env_config | env_override)
    )
    batch = []
    for _ in range(loops):
        batch.append(
            [
                x.id
                for x in list(islice(testing_orderbooks, 0, total_test_orderbooks + 1))
            ]
        )
    assert not all(element == batch[0] for element in batch[1:])

    # Ensure none of the training orderbooks are also test orderbooks
    training_orderbooks, testing_orderbooks = BackTester.load(
        env_config=SimEnv0_Config(**env_config | env_override)
    )

    # Get all testing orderbook ids into a set
    testing_orderbook_ids = {
        x.id for x in list(islice(testing_orderbooks, 0, total_test_orderbooks))
    }
    assert len(testing_orderbook_ids) == total_test_orderbooks

    for ob in training_orderbooks:
        assert ob.id not in testing_orderbook_ids


@pytest.mark.skip
def test_backtester_switch_0():
    # Only test orderbooks

    episode_length = 1500
    episodes = 20
    env_override: dict[str, Any] = {
        "episode_length": episode_length,
        "episodes": episodes,
        "front_load": True,
    }

    backtester = BackTester(env_config=SimEnv0_Config(**env_config | env_override))

    # Get a separate copy of the test orderbooks
    _, testing_orderbooks = BackTester.load(
        env_config=SimEnv0_Config(**env_config | env_override)
    )

    total_test_orderbooks = episodes * episode_length

    for i in range(5):
        with backtester.env2test():
            # Ensure that the set of returned orderbooks are as expected
            for i, ob in enumerate(
                list(islice(testing_orderbooks, 0, total_test_orderbooks))
            ):
                step_id = backtester.step(np.array([-7.1, -7.1]))[3]["ob_id"]

                if ob.id != step_id:
                    logging.warning(f"{i}. {ob.id=}, {step_id=}")
                    raise SystemExit


@pytest.mark.skip
def test_backtester_switch_1():
    # Only training orderbooks

    episode_length = 1000
    episodes = 10
    env_override: dict[str, Any] = {
        "episode_length": episode_length,
        "episodes": episodes,
        "front_load": True,
    }

    backtester = BackTester(env_config=SimEnv0_Config(**env_config | env_override))

    # Get a separate copy of the training orderbooks
    training_orderbooks, _ = BackTester.load(
        env_config=SimEnv0_Config(**env_config | env_override)
    )

    next(training_orderbooks)  # Skip the first one

    for i in range(5):
        with backtester.env2test():
            pass  # Don't bother with test orderbooks

        # Training orderbooks
        for i, ob in enumerate(list(islice(training_orderbooks, 0, 10000))):
            step_id = backtester.step(np.array([-7.1, -7.1]))[3]["ob_id"]
            assert ob.id == step_id


@pytest.mark.skip
def test_backtester_switch_2():
    # Both testing and training orderbooks

    episode_length = 200
    episodes = 30
    env_override: dict[str, Any] = {
        "episode_length": episode_length,
        "episodes": episodes,
        "front_load": True,
    }

    backtester = BackTester(env_config=SimEnv0_Config(**env_config | env_override))

    # Get a separate copy of the test orderbooks
    training_orderbooks, testing_orderbooks = BackTester.load(
        env_config=SimEnv0_Config(**env_config | env_override)
    )

    next(training_orderbooks)  # Skip the first one

    total_test_orderbooks = episodes * episode_length

    for i in range(5):
        with backtester.env2test():
            # Ensure that the set of returned orderbooks are as expected
            for i, ob in enumerate(
                list(islice(testing_orderbooks, 0, total_test_orderbooks))
            ):
                step_id = backtester.step(np.array([-7.1, -7.1]))[3]["ob_id"]
                assert ob.id == step_id

        # Do the same again, but this time on the training orderbooks
        for i, ob in enumerate(list(islice(training_orderbooks, 0, 10000))):
            step_id = backtester.step(np.array([-7.1, -7.1]))[3]["ob_id"]
            logging.debug(f"{i=}, {ob.id=}, {step_id=}, {ob.id==step_id}")
            assert ob.id == step_id


@pytest.mark.skip
def test_backtester_switch_3():
    # Both testing and training orderbooks, front load is False

    episode_length = 200
    episodes = 30
    env_override: dict[str, Any] = {
        "episode_length": episode_length,
        "episodes": episodes,
        "front_load": False,
    }

    backtester = BackTester(env_config=SimEnv0_Config(**env_config | env_override))

    # Get a separate copy of the test orderbooks
    training_orderbooks, testing_orderbooks = BackTester.load(
        env_config=SimEnv0_Config(**env_config | env_override)
    )

    next(training_orderbooks)  # Skip the first one

    total_test_orderbooks = episodes * episode_length

    for i in range(5):
        with backtester.env2test():
            # Ensure that the set of returned orderbooks are as expected
            for i, ob in enumerate(
                list(islice(testing_orderbooks, 0, total_test_orderbooks))
            ):
                step_id = backtester.step(np.array([-7.1, -7.1]))[3]["ob_id"]
                assert ob.id == step_id

        # Do the same again, but this time on training orderbooks
        for i, ob in enumerate(list(islice(training_orderbooks, 0, 10000))):
            step_id = backtester.step(np.array([-7.1, -7.1]))[3]["ob_id"]
            logging.debug(f"{i=}, {ob.id=}, {step_id=}, {ob.id==step_id}")
            assert ob.id == step_id
