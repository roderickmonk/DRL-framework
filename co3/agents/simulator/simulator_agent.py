import importlib
import itertools as it
import logging
import os
from collections import deque, namedtuple
from contextlib import contextmanager
from pprint import pprint
from random import choices
from typing import Any, Tuple

import numpy as np
import torch
from constants import DEVICE
from devtools import debug as debug2
from networks.fully_connected import FullyConnected
from pymongo import MongoClient
from rewards_csv import SimulationRewardsCsv
from sentient_util.get_pdf import get_pdf
from util import apply_depth

torch.set_printoptions(precision=10, sci_mode=True)


class Agent:
    def __init__(self, *, config, env):

        self.config = config
        self.env = env
        self.depth = config.agent.trader_config.depth

        with MongoClient(host=os.environ["MONGODB"]) as mongo_client:

            pdf = get_pdf(
                mongo_client["configuration"]["PDFs"], config.agent.trader_config.pdf
            )

        self.trader = importlib.import_module(
            self.config.agent.trader_config.trader.lower()
        ).Trader(  # type:ignore
            trader_config=self.config.agent.trader_config,
            pdf_x=pdf["x"],
            pdf_y=pdf["y"],
        )

        self.rewards_csv = SimulationRewardsCsv(config=self.config)

    def predicted(self, state, action):
        return 0.0, 0.0

    def select_action(self, state: Tuple[np.ndarray, np.ndarray]):
        return self.trader.compute_orders(
            apply_depth(self.depth, state[0]), apply_depth(self.depth, state[1])
        )

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.rewards_csv.close()
        logging.warning(f"__exit__ traceback: {args=}")

    def __call__(self):
        """"""

        def csv_write():

            buy_match = ""
            buy_Rs = ""
            buy_Qs = ""
            sell_match = ""
            sell_Rs = ""
            sell_Qs = ""

            if step_info["matching"]:
                (
                    buy_match,
                    buy_trading_activity,
                    sell_match,
                    sell_trading_activity,
                ) = step_info["matching"]
                buy_Rs = buy_trading_activity.trade_Rs
                buy_Qs = buy_trading_activity.trade_Qs
                sell_Rs = sell_trading_activity.trade_Rs
                sell_Qs = sell_trading_activity.trade_Qs

            self.rewards_csv(
                ts=step_info["ts"].isoformat(),
                episode=episode,
                step=step_info["step"],
                reward=reward,
                buy_rate=action[0],
                sell_rate=action[1],
                best_buy=step_info["best_buy"],
                best_sell=step_info["best_sell"],
                delta_funds=step_info["delta_funds"],
                delta_inventory=step_info["delta_inventory"],
                buy_match=buy_match,
                buy_Rs=buy_Rs,
                buy_Qs=buy_Qs,
                sell_match=sell_match,
                sell_Rs=sell_Rs,
                sell_Qs=sell_Qs,
            )

        for episode in range(self.config.agent.episodes):

            episode_reward = 0

            state = self.env.reset()

            done = False
            while not done:

                action = self.select_action(state)

                next_state, reward, done, step_info = self.env.step(action)

                if step_info is None:
                    raise RuntimeError("Simulator Failure")

                # debug2(step_info)

                # if step_info["matching"]:
                #     csv_write()

                episode_reward += reward

                if self.config.misc.generate_csv:
                    csv_write()

                state = next_state
