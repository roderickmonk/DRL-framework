import logging
import os

from dateutil import parser
import numpy as np
import pandas as pd
import pymongo
from typing import Any

np.set_printoptions(precision=10, floatmode="fixed", edgeitems=8)  # type:ignore

_log = logging.debug


class TradeHistory:
    """"""

    vps: Any
    pf: Any
    bid: Any
    ask: Any

    @staticmethod
    def getInstance(*, config=None):
        if not hasattr(TradeHistory, "__instance"):
            setattr(TradeHistory, "__instance", TradeHistory(config=config))
            _log("Trade History Instance Created")

        assert hasattr(TradeHistory, "__instance")
        assert hasattr(TradeHistory, "vps")
        assert hasattr(TradeHistory, "pf")
        return TradeHistory.__instance

    def __init__(self, *, config):

        _log(f"Trade History....loading")

        with pymongo.MongoClient(host=os.environ["MONGODB"]) as client:

            db = client["history"]

            self.config = config

            assert (
                TradeHistory.__instance == None
            ), "Only one instance of TradeHistory permitted"

            TradeHistory.__instance = self

            th_collection = db["trades"]

            _log(f"{parser.parse(config.start_range)=}")
            _log(f"{parser.parse(config.end_range)=}")

            trades_raw = list(
                th_collection.find(
                    {
                        "$and": [
                            {"e": config.envId},
                            {"x": config.exchange},
                            {"m": config.market},
                            {
                                "ts": {
                                    "$gte": parser.parse(config.start_range),
                                    "$lt": parser.parse(config.end_range),
                                }
                            },
                        ]
                    },
                    no_cursor_timeout=True,
                )
            )

            trades = pd.DataFrame(trades_raw)

            _log(f"trades\n{trades}")
            _log(f"trades.columns\n{trades.columns}")

            _log(trades["buy"])

            # get the trade size in base and remove those of negligible size
            # limit the dataset to a fixed time range
            trades["t"] = trades.r * trades.q
            trades = trades[trades.t > 0.0005]
            trades = trades[  # type:ignore
                np.logical_and(
                    trades.ts >= config.start_range, trades.ts < config.end_range
                )
            ]

            trades["ts_int"] = trades.ts.astype(np.int64) / 1000000
            trades["seconds"] = np.int64(trades.ts_int / 1000)

            if config.evaluate:
                TradeHistory.vps = trades.groupby([trades["seconds"]])["t"].sum()
                TradeHistory.pf = self.get_profit_factor(TradeHistory.vps)

                """
                self.vps = pd.DataFrame(
                    trades.groupby([trades["seconds"], trades["buy"]])["t"].sum()
                )
                
                logging.debug(f"type(self.vps)\n{type(self.vps)}")
                logging.debug(f"self.vps.describe()\n{self.vps.describe()}")
                logging.debug(f"\nself.vps.index\n{self.vps.index}")
                logging.debug(f"\nself.vps.columns\n{self.vps.columns}")
                logging.debug(f"\nself.vps.values\n{self.vps.values}")
                logging.debug(f"vps\n{self.vps}")
                """

            else:
                TradeHistory.bid = self.vps.query("buy == True").reset_index(
                    level=1, drop=True
                )["t"]
                TradeHistory.ask = self.vps.query("buy == False").reset_index(
                    level=1, drop=True
                )["t"]

                logging.debug(f"len(self.bid): {len(self.bid.index)}")
                logging.debug(f"len(self.ask): {len(self.ask.index)}")
                logging.debug(f"type(self.ask): {type(self.ask)}")
                logging.debug(f"self.ask)\n{self.ask}")

                TradeHistory.bid_pf = self.get_profit_factor(TradeHistory.bid)
                TradeHistory.ask_pf = self.get_profit_factor(TradeHistory.ask)

            logging.debug(f"Trade History....loaded")

    def get_profit_factor(self, vps):
        """
        Profit factor calibrates the relationship between order depth and reward.
        It models the fact that a deeper order implies a more profitable price.
        """

        tsizehist = np.histogram(np.log10(vps), bins=200)

        mids = 10 ** ((tsizehist[1][:-1] + tsizehist[1][1:]) / 2)  # type:ignore
        density = tsizehist[0] / sum(tsizehist[0])  # type:ignore

        expected_fill_sizes = np.array(
            [
                self.expectedFill(mids, density, order_depth)
                for order_depth in self.config.order_depths
            ]
        )

        return 1 / expected_fill_sizes

        """
            plt.plot(config.order_depths, self.pf)
            plt.savefig('pf.png')
            plt.close()
        """

    def expectedFill(self, mids, density, order_depth):
        """
        Calculate the expected fill size for a specific order depth,
        ql, and particular trade size statistics
        """

        fill_sizes = np.maximum(0, mids - order_depth)
        fill_sizes = np.minimum(fill_sizes, self.config.ql)

        expected_fill_size = sum(fill_sizes * density)
        return expected_fill_size

