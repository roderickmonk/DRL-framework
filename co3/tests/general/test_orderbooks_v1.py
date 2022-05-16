import warnings

warnings.filterwarnings("ignore", message="CUDA initialization")

import logging
import os
from datetime import timedelta

from sentient_util import cfg
import numpy as np
import pytest
import util
from dateutil.parser import parse
from sentient_util.orderbooks import Orderbooks
from sentient_util.orderbooks_v1 import OrderbooksV1
from pymongo import MongoClient
from sentient_util.pydantic_config import (
    OrderbookQuery,
    OrderbookWithTrades,
    SimEnv0_Config,
    SimEnv1_Config,
    Trade,
)

primary_mongo = MongoClient(os.environ["MONGODB"])
history_db = primary_mongo.history
ob_collection = history_db.orderbooks


@pytest.fixture()
def delete_test_orderbooks():

    primary_mongo = MongoClient(os.environ["MONGODB"])

    history_db = primary_mongo.history
    ob_collection = history_db.orderbooks

    ob_collection.delete_many({"e": int(99)})


buy_ref = [
    [10, 100],
    [9, 101],
    [8, 102],
    [7, 103],
    [6, 104],
]

sell_ref = [
    [11, 100],
    [12, 101],
    [13, 102],
    [14, 103],
    [15, 104],
]


def insert_delta(*, ts: str, buy: list, sell: list):

    ob_collection.insert_one(
        {
            "V": "V",
            "e": int(99),
            "x": "test-exchange",
            "m": "base-quote",
            "ts": parse(ts),
            "S": int(0),
            "s": False,
            "buy": buy,
            "sell": sell,
        }
    )


# @pytest.mark.skip
# def test_v1_orderbook_snapshot_found_immediately(delete_test_orderbooks):

#     ob_collection.insert_one(
#         {
#             "V": "V",
#             "e": int(99),
#             "x": "test-exchange",
#             "m": "base-quote",
#             "ts": parse("2019-01-01T00:00:00"),
#             "N": int(123),
#             "s": True,
#             "buy": buy_ref,
#             "sell": sell_ref,
#         }
#     )

#     ob_collection.insert_one(
#         {
#             "V": "V",
#             "e": int(99),
#             "x": "test-exchange",
#             "m": "base-quote",
#             "ts": parse("2019-01-01T01:00:00"),
#             "N": int(124),
#             "s": True,
#             "buy": buy_ref,
#             "sell": sell_ref,
#         }
#     )

#     orderbooks = Orderbooks(
#         primary_mongo=primary_mongo,
#         query=OrderbookQuery(
#             **{
#                 "envId": int(99),
#                 "exchange": "test-exchange",
#                 "market": "base-quote",
#                 "start_range": "2019-01-01T00:00:00",
#                 "end_range": "2019-01-01T02:00:00",
#             }
#         ),
#         metadata_only=False,
#     ).next()

#     next(orderbooks)


def test_v1_orderbook_snapshot_missing(delete_test_orderbooks):

    try:

        query = OrderbookQuery(
            **{
                "envId": int(99),
                "exchange": "test-exchange",
                "market": "base-quote",
                "depth": 12,
                "start_range": "2019-01-01T00:00:00",
                "end_range": "2019-01-01T00:00:01",
            }
        )

        orderbooks = Orderbooks(
            primary_mongo=primary_mongo,
            query=query,
            metadata_only=False,
        )
        primary_orderbooks = primary_mongo["history"]["orderbooks"]
        snapshots = orderbooks.snapshot_cursor(
            primary_orderbooks,
            query,
            orderbooks_class=OrderbooksV1,
        )

        next(snapshots)

        assert False, "Unexpected Orderbook Snapshot Found"

    except StopIteration as msg:
        logging.debug(f"{msg=}")
        pass


def test_v1_orderbook_snapshot_found_later(delete_test_orderbooks):

    try:

        ob_collection.insert_one(
            {
                "V": "V",
                "e": int(99),
                "x": "test-exchange",
                "m": "base-quote",
                "ts": parse("2019-01-01T04:00:00"),
                "S": int(0),
                "s": True,
                "buy": buy_ref,
                "sell": sell_ref,
            }
        )

        ob_collection.insert_one(
            {
                "V": "V",
                "e": int(99),
                "x": "test-exchange",
                "m": "base-quote",
                "ts": parse("2019-01-01T05:00:00"),
                "S": int(0),
                "s": True,
                "buy": buy_ref,
                "sell": sell_ref,
            }
        )

        query = OrderbookQuery(
            **{
                "envId": int(99),
                "exchange": "test-exchange",
                "market": "base-quote",
                "depth": 12,
                "start_range": parse("2019-01-01T08:00:00"),
                "end_range": parse("2019-01-01T09:00:00"),
            }
        )

        orderbooks = Orderbooks(
            primary_mongo=primary_mongo,
            query=query,
            metadata_only=False,
        )
        primary_orderbooks = primary_mongo["history"]["orderbooks"]
        snapshots = orderbooks.snapshot_cursor(
            primary_orderbooks,
            query,
            orderbooks_class=OrderbooksV1,
        )

        next(snapshots)

        assert False, "Inappropriate Snapshot Found"

    except StopIteration as msg:
        logging.debug(f"{msg=}")


# @pytest.mark.skip
# def test_v1_orderbook_next_0_deltas(delete_test_orderbooks):

#     try:
#         ob_collection.insert_one(
#             {
#                 "V": "V",
#                 "e": int(99),
#                 "x": "test-exchange",
#                 "m": "base-quote",
#                 "ts": parse("2019-01-01T00:00:00"),
#                 "N": int(0),
#                 "s": True,
#                 "buy": buy_ref,
#                 "sell": sell_ref,
#             }
#         )

#         ob_collection.insert_one(
#             {
#                 "V": "V",
#                 "e": int(99),
#                 "x": "test-exchange",
#                 "m": "base-quote",
#                 "ts": parse("2019-01-01T00:00:01"),
#                 "N": int(0),
#                 "s": True,
#                 "buy": buy_ref,
#                 "sell": sell_ref,
#             }
#         )

#         orderbooks = Orderbooks(
#             primary_mongo=primary_mongo,
#             query=OrderbookQuery(
#                 **{
#                     "envId": int(99),
#                     "exchange": "test-exchange",
#                     "market": "base-quote",
#                     "depth": 12,
#                     "start_range": parse("2019-01-01T00:00:00"),
#                     "end_range": parse("2019-01-01T00:00:02"),
#                 }
#             ),
#             metadata_only=False,
#         ).next()

#         next(orderbooks)

#     except StopIteration:
#         assert False


# @pytest.mark.skip
# def test_v1_orderbook_1_late_delta_then_next(delete_test_orderbooks):

#     try:
#         ob_collection.insert_one(
#             {
#                 "V": "V",
#                 "e": int(99),
#                 "x": "test-exchange",
#                 "m": "base-quote",
#                 "ts": parse("2019-01-01T01:00:00"),
#                 "N": int(123),
#                 "s": True,
#                 "buy": buy_ref,
#                 "sell": sell_ref,
#             }
#         )

#         # This delta is too late to be considered
#         insert_delta(
#             ts="2019-01-01T00:00:00",
#             buy=[[2, 10, 200]],
#             sell=[[2, 11, 200]],
#         )

#         buy_ref2 = [[i * 2 for i in row] for row in buy_ref]
#         sell_ref2 = [[i * 2 for i in row] for row in sell_ref]

#         ob_collection.insert_one(
#             {
#                 "V": "V",
#                 "e": int(99),
#                 "x": "test-exchange",
#                 "m": "base-quote",
#                 "ts": parse("2019-01-01T01:01:00"),
#                 "N": int(124),
#                 "s": True,
#                 "buy": buy_ref2,
#                 "sell": sell_ref2,
#             }
#         )

#         orderbooks = Orderbooks(
#             primary_mongo=primary_mongo,
#             query=OrderbookQuery(
#                 **{
#                     "envId": int(99),
#                     "exchange": "test-exchange",
#                     "market": "base-quote",
#                     "depth": 10000,
#                     "start_range": "2019-01-01T01:00:00",
#                     "end_range": "2019-01-01T02:00:00",
#                 }
#             ),
#             metadata_only=False,
#         ).next()

#         orderbook = next(orderbooks)

#         assert orderbook["S"] == 123
#         assert np.array_equal(orderbook["buy"], np.array(buy_ref))
#         assert np.array_equal(orderbook["sell"], np.array(sell_ref))

#         orderbook = next(orderbooks)
#         assert orderbook["S"] == 124
#         assert orderbook["ts"] == parse("2019-01-01T01:01:00")
#         assert np.array_equal(orderbook["buy"], np.array(buy_ref2))
#         assert np.array_equal(orderbook["sell"], np.array(sell_ref2))

#     except StopIteration:
#         assert False


def test_v1_orderbook_1_delta(delete_test_orderbooks):

    # snapshot 1
    ob_collection.insert_one(
        {
            "V": "V",
            "e": int(99),
            "x": "test-exchange",
            "m": "base-quote",
            "ts": parse("2019-01-01T00:00:00"),
            "N": int(0),
            "s": True,
            "buy": buy_ref,
            "sell": sell_ref,
        }
    )

    # snapshot 2

    buy_ref2 = [[i * 2 for i in row] for row in buy_ref]
    sell_ref2 = [[i * 2 for i in row] for row in sell_ref]

    ob_collection.insert_one(
        {
            "V": "V",
            "e": int(99),
            "x": "test-exchange",
            "m": "base-quote",
            "ts": parse("2019-01-01T00:00:02"),
            "N": int(2),
            "s": True,
            "buy": buy_ref2,
            "sell": sell_ref2,
        }
    )

    buy_delta = [
        [2, 10, 200],
        [2, 9, 201],
        [2, 8, 202],
        [2, 7, 203],
        [2, 6, 204],
    ]

    sell_delta = [
        [2, 11, 200],
        [2, 12, 201],
        [2, 13, 202],
        [2, 14, 203],
        [2, 15, 204],
    ]

    ob_collection.insert_one(
        {
            "V": "V",
            "e": int(99),
            "x": "test-exchange",
            "m": "base-quote",
            "ts": parse("2019-01-01T00:00:02"),
            "N": int(1),
            "s": False,
            "buy": buy_delta,
            "sell": sell_delta,
        }
    )

    query = OrderbookQuery(
        **{
            "envId": int(99),
            "exchange": "test-exchange",
            "market": "base-quote",
            "depth": 100000,
            "start_range": parse("2019-01-01T00:00:00"),
            "end_range": parse("2019-01-01T01:00:00"),
        }
    )

    orderbooks = Orderbooks(
        primary_mongo=primary_mongo,
        query=query,
        metadata_only=False,
    )
    primary_orderbooks = primary_mongo["history"]["orderbooks"]
    breaks_collection = primary_mongo["derived-history"]["breaks"]
    snapshots = orderbooks.snapshot_cursor(
        primary_orderbooks,
        query,
        orderbooks_class=OrderbooksV1,
    )

    gen = orderbooks.generator(
        primary_orderbooks=primary_orderbooks,
        breaks_collection=breaks_collection,
        orderbooks_class=OrderbooksV1,
        snapshots=snapshots,
        query=query,
        metadata_only=False,
    )

    orderbook = next(gen)

    assert np.array_equal(orderbook["buy"], np.array(buy_ref))
    assert np.array_equal(orderbook["sell"], np.array(sell_ref))

    logging.debug(f"{orderbook['buy']=}")
    logging.debug(f"{orderbook['sell']=}")

    buy_expected = [
        [10, 200],
        [9, 201],
        [8, 202],
        [7, 203],
        [6, 204],
    ]

    sell_expected = [
        [11, 200],
        [12, 201],
        [13, 202],
        [14, 203],
        [15, 204],
    ]

    orderbook = next(gen)
    assert np.array_equal(orderbook["buy"], np.array(buy_expected))
    assert np.array_equal(orderbook["sell"], np.array(sell_expected))


def test_v1_apply_depth_1():

    orderbook = np.array(
        [
            [10, 100],
            [9, 100],
            [8, 100],
            [7, 100],
            [6, 100],
        ],
        dtype=float,
    )

    ob = util.apply_depth(2000, orderbook)

    logging.debug(f"{ob=}")
    assert np.array_equal(
        ob,
        np.array(
            [
                [10, 100],
                [9, 100],
                [8, 100],
            ]
        ),
    )


def test_v1_apply_depth_2():
    # Subset of the OB returned

    orderbook = np.array(
        [
            [10, 100],
            [9, 100],
            [8, 100],
            [7, 100],
            [6, 100],
        ],
        dtype=float,
    )

    ob = util.apply_depth(2000, orderbook)

    assert np.array_equal(
        ob,
        np.array(
            [
                [10, 100],
                [9, 100],
                [8, 100],
            ],
            dtype=float,
        ),
    )


def test_v1_apply_depth_3():

    # First element exactly equals depth

    orderbook = np.array(
        [
            [10, 100],
            [9, 100],
            [8, 100],
            [7, 100],
            [6, 100],
        ],
        dtype=float,
    )

    ob = util.apply_depth(1000, orderbook)
    assert np.array_equal(
        ob,
        np.array(
            [
                [10, 100],
            ],
            dtype=float,
        ),
    )


def test_v1_apply_depth_4():

    # First element > depth

    orderbook = np.array(
        [
            [10, 100],
            [9, 100],
            [8, 100],
            [7, 100],
            [6, 100],
        ],
        dtype=float,
    )

    ob = util.apply_depth(999, orderbook)
    assert np.array_equal(
        ob,
        np.array(
            [
                [10, 100],
            ],
            dtype=float,
        ),
    )


def test_v1_apply_depth_5():

    # Very large depth - return the entire OB

    orderbook = np.array(
        [
            [10, 100],
            [9, 100],
            [8, 100],
            [7, 100],
            [6, 100],
        ],
        dtype=float,
    )

    ob = util.apply_depth(100000, orderbook)
    assert np.array_equal(ob, np.array(orderbook, dtype=float))
