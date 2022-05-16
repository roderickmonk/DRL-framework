import warnings

warnings.filterwarnings("ignore", category=UserWarning)
import logging
import os
from copy import deepcopy

import numpy as np
from sentient_util import cfg
from env_modules.orderbook_vectors import OrderbookVectors
from pymongo import MongoClient

logging.basicConfig(
    format="[%(levelname)-5s] %(message)s", level=logging.INFO, datefmt=""
)

primary_mongo = MongoClient(os.environ["MONGODB"])
history_db = primary_mongo.history
ob_collection = history_db.orderbooks

np.set_printoptions(precision=14, floatmode="fixed", edgeitems=25)

base_config = {
    "envId": 99,
    "exchange": "dummy",
    "market": "dummy",
}


def test_filter_small_entries_0():

    env_config = cfg.ConfigObj(
        base_config
        | {"start_range": "2020-01-01", "end_range": "2020-01-02", "n": 4, "k": 1.5,}
    )
    result = OrderbookVectors(
        primary_mongo=primary_mongo, env_config=deepcopy(env_config)
    ).filter_small_entries(np.array([[1, 100], [2, 200], [3, 300], [4, 1]]))

    expected = np.array([[1, 100], [2, 200], [3, 301]])

    assert np.array_equal(result, expected)


def test_filter_small_entries_1():

    env_config = cfg.ConfigObj(
        base_config
        | {"start_range": "2020-01-01", "end_range": "2020-01-02", "n": 4, "k": 1.5}
    )
    result = OrderbookVectors(
        primary_mongo=primary_mongo, env_config=deepcopy(env_config)
    ).filter_small_entries(np.array([[1, 100], [2, 200], [3, 1], [4, 400]]))

    expected = np.array([[1, 100], [2, 201], [4, 400]])

    assert np.array_equal(result, expected)


def test_filter_small_entries_2():

    env_config = cfg.ConfigObj(
        base_config
        | {"start_range": "2020-01-01", "end_range": "2020-01-02", "n": 4, "k": 1.5}
    )
    result = OrderbookVectors(
        primary_mongo=primary_mongo, env_config=deepcopy(env_config)
    ).filter_small_entries(np.array([[1, 100], [2, 1], [3, 300], [4, 400]]))

    expected = np.array([[1, 101], [3, 300], [4, 400]])

    assert np.array_equal(result, expected)


def test_filter_small_entries_3():

    env_config = cfg.ConfigObj(
        base_config
        | {"start_range": "2020-01-01", "end_range": "2020-01-02", "n": 4, "k": 1.5}
    )
    result = OrderbookVectors(
        primary_mongo=primary_mongo, env_config=deepcopy(env_config)
    ).filter_small_entries(np.array([[1, 100], [2, 200], [3, 0.5], [4, 0.5]]))

    expected = np.array([[1, 100], [2, 201]])

    assert np.array_equal(result, expected)


def test_filter_small_entries_4():

    env_config = cfg.ConfigObj(
        base_config
        | {"start_range": "2020-01-01", "end_range": "2020-01-02", "n": 4, "k": 1.5}
    )
    result = OrderbookVectors(
        primary_mongo=primary_mongo, env_config=deepcopy(env_config)
    ).filter_small_entries(np.array([[1, 100], [2, 1], [3, 300], [4, 1]]))

    expected = np.array([[1, 101], [3, 301]])

    assert np.array_equal(result, expected)


def test_filter_small_entries_5():

    env_config = cfg.ConfigObj(
        base_config
        | {"start_range": "2020-01-01", "end_range": "2020-01-02", "n": 4, "k": 1.5}
    )
    result = OrderbookVectors(
        primary_mongo=primary_mongo, env_config=deepcopy(env_config)
    ).filter_small_entries(np.array([[1, 100], [2, 200], [3, 1], [4, 1]]))

    expected = np.array([[1, 100], [2, 200], [3, 2]])

    assert np.array_equal(result, expected)


def test_filter_small_entries_6():

    env_config = cfg.ConfigObj(
        base_config
        | {"start_range": "2020-01-01", "end_range": "2020-01-02", "n": 4, "k": 10}
    )
    result = OrderbookVectors(
        primary_mongo=primary_mongo, env_config=deepcopy(env_config)
    ).filter_small_entries(np.array([[1, 100], [2, 1], [3, 1], [4, 1]]))

    expected = np.array([[1, 103]])

    assert np.array_equal(result, expected)


def test_filter_small_entries_7():

    env_config = cfg.ConfigObj(
        base_config
        | {"start_range": "2020-01-01", "end_range": "2020-01-02", "n": 4, "k": 10}
    )

    result = OrderbookVectors(
        primary_mongo=primary_mongo, env_config=env_config
    ).filter_small_entries(np.array([[1, 1], [2, 1], [3, 1], [4, 1]]))

    expected = np.array([[1, 4]])

    assert np.array_equal(result, expected)


def test_filter_small_entries_8():

    env_config = cfg.ConfigObj(
        base_config
        | {"start_range": "2020-01-01", "end_range": "2020-01-02", "n": 4, "k": 3}
    )

    result = OrderbookVectors(
        primary_mongo=primary_mongo, env_config=env_config
    ).filter_small_entries(np.array([[1, 1], [2, 1], [3, 1], [4, 1]]))

    expected = np.array([[1, 4]])

    assert np.array_equal(result, expected)


def test_filter_small_entries_9():

    env_config = cfg.ConfigObj(
        base_config
        | {"start_range": "2020-01-01", "end_range": "2020-01-02", "n": 4, "k": 3}
    )

    result = OrderbookVectors(
        primary_mongo=primary_mongo, env_config=env_config
    ).filter_small_entries(np.array([[1, 1], [2, 1], [3, 1], [4, 1], [5, 1]]))

    expected = np.array([[1, 1], [2, 4]])

    assert np.array_equal(result, expected)


def test_first_truncation_0():

    env_config = cfg.ConfigObj(
        base_config
        | {"start_range": "2020-01-01", "end_range": "2020-01-02", "n": 4, "k": 3}
    )
    result = OrderbookVectors(
        primary_mongo=primary_mongo, env_config=env_config
    ).first_truncation(
        np.array(
            [
                [1, 1],
                [2, 1],
                [3, 1],
                [4, 1],
                [5, 1],
                [6, 4],
                [7, 4],
                [8, 4],
                [9, 4],
                [10, 4],
            ]
        )
    )

    expected = np.array(
        [[1, 1], [2, 1], [3, 1], [4, 1], [5, 1], [6, 4], [7, 4], [8, 4], [9, 4],]
    )

    assert np.array_equal(result, expected)


def test_first_truncation_1():

    env_config = cfg.ConfigObj(
        base_config
        | {"start_range": "2020-01-01", "end_range": "2020-01-02", "n": 4, "k": 3}
    )
    result = OrderbookVectors(
        primary_mongo=primary_mongo, env_config=env_config
    ).first_truncation(
        np.array(
            [
                [1, 3],
                [2, 3],
                [3, 3],
                [4, 3],
                [5, 3],
                [6, 3],
                [7, 3],
                [8, 3],
                [9, 3],
                [10, 3],
            ]
        )
    )

    expected = np.array(
        [
            [1, 3],
            [2, 3],
            [3, 3],
            [4, 3],
            [5, 3],
            [6, 3],
            [7, 3],
            [8, 3],
            [9, 3],
            [10, 3],
        ]
    )

    assert np.array_equal(result, expected)


def test_first_truncation_2():

    env_config = cfg.ConfigObj(
        base_config
        | {"start_range": "2020-01-01", "end_range": "2020-01-02", "n": 4, "k": 2}
    )
    result = OrderbookVectors(
        primary_mongo=primary_mongo, env_config=env_config
    ).first_truncation(
        np.array(
            [
                [1, 3],
                [2, 3],
                [3, 3],
                [4, 3],
                [5, 3],
                [6, 3],
                [7, 3],
                [8, 3],
                [9, 3],
                [10, 3],
            ]
        )
    )

    expected = np.array([[1, 3], [2, 3], [3, 3], [4, 3],])

    assert np.array_equal(result, expected)


def test_create_sub_vector_0():

    env_config = cfg.ConfigObj(
        base_config
        | {"start_range": "2020-01-01", "end_range": "2020-01-02", "n": 4, "k": 2}
    )
    result = OrderbookVectors(
        primary_mongo=primary_mongo, env_config=env_config
    ).create_sub_vector(
        mid_price=50,
        OB=np.array(
            [
                [1, 3],
                [2, 3],
                [3, 3],
                [4, 3],
                [5, 3],
                [6, 3],
                [7, 3],
                [8, 3],
                [9, 3],
                [10, 3],
            ]
        ),
    )

    expected = np.array(
        [
            [0.98000000000000, 0.00000000000000],
            [0.96000000000000, 3.00000000000000],
            [0.94000000000000, 9.00000000000000],
            [0.92000000000000, 18.00000000000000],
        ]
    )

    # logging.info(f"{result=}")

    assert np.array_equal(result, expected)


def test_create_sub_vector_1():

    env_config = cfg.ConfigObj(
        base_config
        | {"start_range": "2020-01-01", "end_range": "2020-01-02", "n": 5, "k": 2}
    )

    result = OrderbookVectors(
        primary_mongo=primary_mongo, env_config=env_config
    ).create_sub_vector(
        mid_price=50,
        OB=np.array(
            [
                [1, 3],
                [2, 3],
                [3, 3],
                [4, 3],
                [5, 3],
                [6, 3],
                [7, 3],
                [8, 3],
                [9, 3],
                [10, 3],
            ]
        ),
    )

    expected = np.array(
        [
            [0.98000000000000, 0.00000000000000],
            [0.96000000000000, 3.00000000000000],
            [0.94000000000000, 9.00000000000000],
            [0.92000000000000, 18.00000000000000],
            [0.90000000000000, 30.00000000000000],
        ]
    )

    # logging.info(f"{result=}")

    assert np.array_equal(result, expected)


def test_create_sub_vector_2():

    env_config = cfg.ConfigObj(
        base_config
        | {"start_range": "2020-01-01", "end_range": "2020-01-02", "n": 3, "k": 2}
    )

    result = OrderbookVectors(
        primary_mongo=primary_mongo, env_config=env_config
    ).create_sub_vector(
        mid_price=50,
        OB=np.array(
            [
                [1, 3],
                [2, 3],
                [3, 3],
                [4, 3],
                [5, 3],
                [6, 3],
                [7, 3],
                [8, 3],
                [9, 3],
                [10, 3],
            ]
        ),
    )

    expected = np.array(
        [
            [0.98000000000000, 0.00000000000000],
            [0.96000000000000, 3.00000000000000],
            [0.94000000000000, 9.00000000000000],
        ]
    )

    # logging.info(f"{result=}")

    assert np.array_equal(result, expected)


def test_create_sub_vector_3():

    env_config = cfg.ConfigObj(
        base_config
        | {"start_range": "2020-01-01", "end_range": "2020-01-02", "n": 3, "k": 100}
    )

    result = OrderbookVectors(
        primary_mongo=primary_mongo, env_config=env_config
    ).create_sub_vector(
        mid_price=1,
        OB=np.array(
            [
                [1, 3],
                [2, 3],
                [3, 3],
                [4, 3],
                [5, 3],
                [6, 3],
                [7, 3],
                [8, 3],
                [9, 3],
                [10, 3],
            ]
        ),
    )

    expected = np.array(
        [
            [0.00000000000000, 0.00000000000000],
            [6.00000000000000, 63.00000000000000],
            [6.00000000000000, 63.00000000000000],
        ]
    )

    # logging.info(f"{result=}")

    assert np.array_equal(result, expected)


def test_create_sub_vector_4():

    env_config = cfg.ConfigObj(
        base_config
        | {"start_range": "2020-01-01", "end_range": "2020-01-02", "n": 5, "k": 15}
    )

    result = OrderbookVectors(
        primary_mongo=primary_mongo, env_config=env_config
    ).create_sub_vector(
        mid_price=1,
        OB=np.array(
            [
                [1, 3],
                [2, 3],
                [3, 3],
                [4, 3],
                [5, 3],
                [6, 3],
                [7, 3],
                [8, 3],
                [9, 3],
                [10, 3],
            ]
        ),
    )

    expected = np.array(
        [
            [0.00000000000000, 0.00000000000000],
            [3.00000000000000, 18.00000000000000],
            [5.00000000000000, 45.00000000000000],
            [6.00000000000000, 63.00000000000000],
            [7.00000000000000, 84.00000000000000],
        ]
    )

    # logging.info(f"{result=}")

    assert np.array_equal(result, expected)


def test_create_sub_vector_5():

    env_config = cfg.ConfigObj(
        base_config
        | {"start_range": "2020-01-01", "end_range": "2020-01-02", "n": 5, "k": 15}
    )

    result = OrderbookVectors(
        primary_mongo=primary_mongo, env_config=env_config
    ).create_sub_vector(
        mid_price=0.5,
        OB=np.array(
            [
                [1, 3],
                [2, 3],
                [3, 3],
                [4, 3],
                [5, 3],
                [6, 3],
                [7, 3],
                [8, 3],
                [9, 3],
                [10, 3],
            ]
        ),
    )

    expected = np.array(
        [
            [1.00000000000000, 0.00000000000000],
            [7.00000000000000, 18.00000000000000],
            [11.00000000000000, 45.00000000000000],
            [13.00000000000000, 63.00000000000000],
            [15.00000000000000, 84.00000000000000],
        ]
    )

    # logging.info(f"{result=}")

    assert np.array_equal(result, expected)


def test_create_sub_vector_6():

    env_config = cfg.ConfigObj(
        base_config
        | {"start_range": "2020-01-01", "end_range": "2020-01-02", "n": 5, "k": 15}
    )

    result = OrderbookVectors(
        primary_mongo=primary_mongo, env_config=env_config
    ).create_sub_vector(
        mid_price=0.5,
        OB=np.array(
            [
                [1, 3],
                [2, 3],
                [3, 3],
                [4, 3],
                [5, 3],
                [6, 3],
                [7, 3],
                [8, 3],
                [9, 3],
                [10, 3],
            ]
        ),
    )

    expected = np.array(
        [
            [1.00000000000000, 0.00000000000000],
            [7.00000000000000, 18.00000000000000],
            [11.00000000000000, 45.00000000000000],
            [13.00000000000000, 63.00000000000000],
            [15.00000000000000, 84.00000000000000],
        ]
    )

    # logging.info(f"{result=}")

    assert np.array_equal(result, expected)


def test_create_sub_vector_7():

    env_config = cfg.ConfigObj(
        base_config
        | {"start_range": "2020-01-01", "end_range": "2020-01-02", "n": 5, "k": 2}
    )

    try:
        OrderbookVectors(
            primary_mongo=primary_mongo, env_config=env_config
        ).create_sub_vector(
            mid_price=0.5,
            OB=np.array(
                [
                    [1, 3],
                    [0.5, 3],  # Not monotonic
                    [3, 3],
                    [4, 3],
                    [5, 3],
                    [6, 3],
                    [7, 3],
                    [8, 3],
                    [9, 3],
                    [10, 3],
                ]
            ),
        )

    except ValueError:
        return

    assert False, "Monotonic test failed"


def test_create_sub_vector_8():

    env_config = cfg.ConfigObj(
        base_config
        | {"start_range": "2020-01-01", "end_range": "2020-01-02", "n": 5, "k": 6}
    )
    result = OrderbookVectors(
        primary_mongo=primary_mongo, env_config=env_config
    ).create_sub_vector(
        mid_price=0.5,
        OB=np.array(
            [
                [1, 3],
                [2, 3],
                [3, 3],
                [4, 3],
                [5, 3],
                [6, 3],
                [7, 3],
                [8, 3],
                [9, 3],
                [10, 3],
            ]
        ),
    )

    expected = np.array(
        [
            [1.00000000000000, 0.00000000000000],
            [5.00000000000000, 9.00000000000000],
            [7.00000000000000, 18.00000000000000],
            [9.00000000000000, 30.00000000000000],
            [11.00000000000000, 45.00000000000000],
        ]
    )

    # logging.info(f"{result=}")

    assert np.array_equal(result, expected)


def test_create_OB_vector_0():

    env_config = cfg.ConfigObj(
        base_config
        | {"start_range": "2020-01-01", "end_range": "2020-01-02", "n": 4, "k": 10}
    )

    OB = cfg.ConfigObj(
        {
            "buy": np.array(
                [[9, 3], [8, 5], [7, 3], [6, 5], [5, 3], [4, 5], [3, 3], [2, 5],]
            ),
            "sell": np.array(
                [
                    [11, 3],
                    [12, 5],
                    [13, 3],
                    [14, 5],
                    [15, 3],
                    [16, 5],
                    [17, 3],
                    [18, 5],
                ]
            ),
        }
    )

    result = OrderbookVectors(
        primary_mongo=primary_mongo, env_config=env_config
    ).create_OB_vector(OB, side="buy")

    expected = np.array(
        [
            0.10000000000000,
            0.20000000000000,
            0.30000000000000,
            0.40000000000000,
            0.00000000000000,
            27.00000000000000,
            67.00000000000000,
            88.00000000000000,
        ]
    )

    # logging.info(f"{result=}")

    assert result.shape == expected.shape
    assert np.array_equal(np.around(result, 10), np.around(expected, 10))


# The only change in the following from the previous is that k has been increased to 30
def test_create_OB_vector_1():

    env_config = cfg.ConfigObj(
        base_config
        | {"start_range": "2020-01-01", "end_range": "2020-01-02", "n": 4, "k": 30}
    )

    OB = cfg.ConfigObj(
        {
            "buy": np.array(
                [[9, 3], [8, 5], [7, 3], [6, 5], [5, 3], [4, 5], [3, 3], [2, 5],]
            ),
            "sell": np.array(
                [
                    [11, 3],
                    [12, 5],
                    [13, 3],
                    [14, 5],
                    [15, 3],
                    [16, 5],
                    [17, 3],
                    [18, 5],
                ]
            ),
        }
    )

    result = OrderbookVectors(
        primary_mongo=primary_mongo, env_config=env_config
    ).create_OB_vector(OB, side="buy")

    expected = np.array(
        [
            1.00000000000000e-01,
            2.00000000000000e-01,
            4.00000000000000e-01,
            6.00000000000000e-01,
            0.00000000000000e00,
            2.70000000000000e01,
            8.80000000000000e01,
            1.33000000000000e02,
        ]
    )

    # logging.info(f"{result=}")

    assert result.shape == expected.shape
    assert np.array_equal(np.around(result, 10), np.around(expected, 10))


def test_create_OB_vector_2():

    env_config = cfg.ConfigObj(
        base_config
        | {"start_range": "2020-01-01", "end_range": "2020-01-02", "n": 4, "k": 4}
    )

    OB = cfg.ConfigObj(
        {
            "buy": np.array(
                [[9, 3], [8, 5], [7, 3], [6, 5], [5, 3], [4, 5], [3, 3], [2, 5],]
            ),
            "sell": np.array(
                [
                    [11, 3],
                    [12, 5],
                    [13, 3],
                    [14, 5],
                    [15, 3],
                    [16, 5],
                    [17, 3],
                    [18, 5],
                ]
            ),
        }
    )

    result = OrderbookVectors(
        primary_mongo=primary_mongo, env_config=env_config
    ).create_OB_vector(OB, side="sell")

    expected = np.array(
        [
            1.00000000000000e-01,
            2.00000000000000e-01,
            3.00000000000000e-01,
            4.00000000000000e-01,
            0.00000000000000e00,
            3.30000000000000e01,
            9.30000000000000e01,
            1.32000000000000e02,
        ]
    )

    # logging.info(f"{result=}")

    assert result.shape == expected.shape
    assert np.array_equal(np.around(result, 10), np.around(expected, 10))


def test_create_OB_vector_3():

    env_config = cfg.ConfigObj(
        base_config
        | {"start_range": "2020-01-01", "end_range": "2020-01-02", "n": 4, "k": 4}
    )

    OB = cfg.ConfigObj(
        {
            "buy": np.array(
                [[9, 3], [8, 5], [7, 3], [6, 5], [5, 3], [4, 5], [3, 3], [2, 5],]
            ),
            "sell": np.array(
                [
                    [11, 3],
                    [12, 5],
                    [13, 3],
                    [14, 5],
                    [15, 3],
                    [16, 5],
                    [17, 3],
                    [18, 5],
                ]
            ),
        }
    )

    result = OrderbookVectors(
        primary_mongo=primary_mongo, env_config=env_config
    ).create_OB_vector(OB, side="ob")

    expected = np.array(
        [
            1.00000000000000e-01,
            2.00000000000000e-01,
            3.00000000000000e-01,
            4.00000000000000e-01,
            0.00000000000000e00,
            2.70000000000000e01,
            6.70000000000000e01,
            8.80000000000000e01,
            1.00000000000000e-01,
            2.00000000000000e-01,
            3.00000000000000e-01,
            4.00000000000000e-01,
            0.00000000000000e00,
            3.30000000000000e01,
            9.30000000000000e01,
            1.32000000000000e02,
        ]
    )

    # logging.info(f"{result=}")

    assert result.shape == expected.shape

    assert np.array_equal(np.around(result, 10), np.around(expected, 10))
