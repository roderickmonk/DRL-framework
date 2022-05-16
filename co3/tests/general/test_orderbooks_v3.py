import warnings

import numpy as np
from sentient_util.orderbooks_v3 import OrderbooksV3

warnings.filterwarnings("ignore", message="CUDA initialization")

import os
from datetime import timedelta

import dateutil.parser
import pytest
from dateutil.parser import parse
from pymongo import MongoClient
from sentient_util import cfg
from sentient_util.orderbooks import Orderbooks

primary_mongo = MongoClient(os.environ["MONGODB"])
history_db = primary_mongo.history
primary_orderbooks = primary_mongo["history"]["BittrexV3Orderbooks"]
breaks_collection = primary_mongo["derived-history"]["breaks"]
from sentient_util.pydantic_config import (
    OrderbookQuery,
    OrderbookWithTrades,
    SimEnv0_Config,
    SimEnv1_Config,
    Trade,
)


@pytest.fixture()
def delete_test_orderbooks():

    primary_mongo = MongoClient(os.environ["MONGODB"])

    history_db = primary_mongo.history
    primary_orderbooks = history_db.BittrexV3Orderbooks

    primary_orderbooks.delete_many({"e": int(99)})


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


def insert_snapshot():

    primary_orderbooks.insert_one(
        {
            "V": "V",
            "e": int(99),
            "x": "test-exchange",
            "m": "base-quote",
            "ts": parse("2025-01-01T00:00:00"),
            "s": True,
            "buy": buy_ref,
            "sell": sell_ref,
            "v": 3,
        }
    )


def insert_delta(*, ts: str, buy: list, sell: list):

    primary_orderbooks.insert_one(
        {
            "V": "V",
            "e": int(99),
            "x": "test-exchange",
            "m": "base-quote",
            "ts": parse(ts),
            "s": False,
            "buy": buy,
            "sell": sell,
            "v": 3,
        }
    )


def test_v3_orderbook_snapshot_found_immediately(delete_test_orderbooks):

    primary_orderbooks.insert_one(
        {
            "V": "V",
            "e": int(99),
            "x": "test-exchange",
            "m": "base-quote",
            "ts": parse("2025-01-01T00:00:00"),
            "s": True,
            "S": 0,
            "buy": buy_ref,
            "sell": sell_ref,
            "v": 3,
        }
    )

    primary_orderbooks.insert_one(
        {
            "V": "V",
            "e": int(99),
            "x": "test-exchange",
            "m": "base-quote",
            "ts": parse("2025-01-01T00:30:00"),
            "s": True,
            "S": 1,
            "buy": buy_ref,
            "sell": sell_ref,
            "v": 3,
        }
    )

    query = OrderbookQuery(
        **{
            "envId": int(99),
            "exchange": "test-exchange",
            "market": "base-quote",
            "start_range": "2025-01-01T00:00:00",
            "end_range": "2025-01-01T02:00:00",
            "v": 3,
        }
    )

    orderbooks = Orderbooks(
        primary_mongo=primary_mongo,
        query=query,
        metadata_only=False,
    )
    snapshots = orderbooks.snapshot_cursor(
        primary_orderbooks,
        query,
        orderbooks_class=OrderbooksV3,
    )

    next(snapshots)


def test_v3_orderbook_snapshot_missing(delete_test_orderbooks):

    try:

        query = OrderbookQuery(
            **{
                "envId": int(99),
                "exchange": "test-exchange",
                "market": "base-quote",
                "start_range": "2026-01-01T00:00:00",
                "end_range": "2026-01-01T00:00:01",
                "v": 3,
            }
        )

        orderbooks = Orderbooks(
            primary_mongo=primary_mongo,
            query=query,
            metadata_only=False,
        )
        snapshots = orderbooks.snapshot_cursor(
            primary_orderbooks,
            query,
            orderbooks_class=OrderbooksV3,
        )

        next(snapshots)

        assert False, "Unexpected Orderbook Snapshot Found"

    except StopIteration as msg:
        pass


def test_v3_orderbook_snapshot_found_later(delete_test_orderbooks):

    primary_orderbooks.insert_one(
        {
            "V": "V",
            "e": int(99),
            "x": "test-exchange",
            "m": "base-quote",
            "ts": parse("2025-01-01T04:00:00"),
            "s": True,
            "S": 0,
            "buy": buy_ref,
            "sell": sell_ref,
            "v": 3,
        }
    )

    primary_orderbooks.insert_one(
        {
            "V": "V",
            "e": int(99),
            "x": "test-exchange",
            "m": "base-quote",
            "ts": parse("2025-01-01T05:00:00"),
            "s": True,
            "S": 1,
            "buy": buy_ref,
            "sell": sell_ref,
            "v": 3,
        }
    )

    query = OrderbookQuery(
        **{
            "envId": int(99),
            "exchange": "test-exchange",
            "market": "base-quote",
            "depth": 12,
            "start_range": "2025-01-01T05:00:00",
            "end_range": "2025-01-01T05:00:01",
            "v": 3,
        }
    )
    orderbooks = Orderbooks(
        primary_mongo=primary_mongo,
        query=query,
        metadata_only=False,
    )
    snapshots = orderbooks.snapshot_cursor(
        primary_orderbooks,
        query,
        orderbooks_class=OrderbooksV3,
    )

    next(snapshots)


def test_v3_orderbook_snapshot_not_found_within_last_3_hours(delete_test_orderbooks):

    try:

        query = OrderbookQuery(
            **{
                "envId": int(99),
                "exchange": "test-exchange",
                "market": "base-quote",
                "start_range": "2025-01-01T03:00:01",
                "end_range": "2025-01-01T03:00:02",
                "v": 3,
            }
        )

        orderbooks = Orderbooks(
            primary_mongo=primary_mongo,
            query=query,
            metadata_only=False,
        )
        snapshots = orderbooks.snapshot_cursor(
            primary_orderbooks,
            query,
            orderbooks_class=OrderbooksV3,
        )

        next(snapshots)

        assert False, "Unexpected Orderbook Snapshot Found"

    except StopIteration:
        pass


def test_v3_orderbook_next_0_deltas(delete_test_orderbooks):

    try:
        primary_orderbooks.insert_one(
            {
                "V": "V",
                "e": int(99),
                "x": "test-exchange",
                "m": "base-quote",
                "ts": parse("2025-01-01T00:00:00"),
                "s": True,
                "S": 1234,
                "buy": buy_ref,
                "sell": sell_ref,
                "v": 3,
            }
        )

        primary_orderbooks.insert_one(
            {
                "V": "V",
                "e": int(99),
                "x": "test-exchange",
                "m": "base-quote",
                "ts": parse("2025-01-01T00:00:01"),
                "s": True,
                "S": 1235,
                "buy": buy_ref,
                "sell": sell_ref,
                "v": 3,
            }
        )

        query = OrderbookQuery(
            **{
                "envId": int(99),
                "exchange": "test-exchange",
                "market": "base-quote",
                "start_range": "2025-01-01T00:00:00",
                "end_range": "2025-01-01T00:00:02",
                "v": 3,
            }
        )

        snapshots = Orderbooks(
            primary_mongo=primary_mongo,
            query=query,
            metadata_only=False,
        ).snapshot_cursor(
            primary_orderbooks,
            query,
            orderbooks_class=OrderbooksV3,
        )

        # Ensure 2 OBs are available
        next(snapshots)
        next(snapshots)

    except StopIteration:
        assert False


def test_v3_orderbook_next_1_deltas(delete_test_orderbooks):

    try:
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

        buy_delta = [[10, 1000], [6, 0]]
        sell_delta = [[15, 1000], [11, 0]]

        buy_ref2 = [[i * 2 for i in row] for row in buy_ref]
        sell_ref2 = [[i * 2 for i in row] for row in sell_ref]

        # snapshot
        primary_orderbooks.insert_one(
            {
                "V": "V",
                "e": int(99),
                "x": "test-exchange",
                "m": "base-quote",
                "ts": parse("2025-01-01T00:00:00"),
                "s": True,
                "S": 1234,
                "buy": buy_ref,
                "sell": sell_ref,
                "v": 3,
            }
        )

        # delta
        primary_orderbooks.insert_one(
            {
                "V": "V",
                "e": int(99),
                "x": "test-exchange",
                "m": "base-quote",
                "ts": parse("2025-01-01T00:00:01"),
                "s": False,
                "S": 1235,
                "buy": buy_delta,
                "sell": sell_delta,
                "v": 3,
            }
        )

        # snapshot
        primary_orderbooks.insert_one(
            {
                "V": "V",
                "e": int(99),
                "x": "test-exchange",
                "m": "base-quote",
                "ts": parse("2025-01-01T00:00:02"),
                "s": True,
                "S": 1236,
                "buy": buy_ref2,
                "sell": sell_ref2,
                "v": 3,
            }
        )

        query = OrderbookQuery(
            **{
                "envId": int(99),
                "exchange": "test-exchange",
                "market": "base-quote",
                "start_range": "2025-01-01T00:00:00",
                "end_range": "2025-01-01T00:00:03",
                "v": 3,
            }
        )

        orderbooks = Orderbooks(
            primary_mongo=primary_mongo,
            query=query,
            metadata_only=False,
        )

        snapshots = orderbooks.snapshot_cursor(
            primary_orderbooks, query, orderbooks_class=OrderbooksV3
        )

        gen = orderbooks.generator(
            primary_orderbooks=primary_orderbooks,
            breaks_collection=breaks_collection,
            orderbooks_class=OrderbooksV3,
            snapshots=snapshots,
            query=query,
            metadata_only=False,
        )

        orderbook = next(gen)
        assert np.array_equal(orderbook["buy"], np.array(buy_ref))
        assert np.array_equal(orderbook["sell"], np.array(sell_ref))

        buy_expected = [[10, 1000], [9, 101], [8, 102], [7, 103]]
        sell_expected = [[12, 101], [13, 102], [14, 103], [15, 1000]]

        orderbook = next(gen)
        assert np.array_equal(orderbook["buy"], np.array(buy_expected))
        assert np.array_equal(orderbook["sell"], np.array(sell_expected))

        orderbook = next(gen)
        assert np.array_equal(orderbook["buy"], np.array(buy_ref2))
        assert np.array_equal(orderbook["sell"], np.array(sell_ref2))

    except StopIteration:
        assert False
