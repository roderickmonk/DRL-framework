#!/usr/bin/env python
import itertools as it
from collections import deque
from dataclasses import dataclass
from itertools import count
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from pydantic_config import ProcessConfig
from torch.utils.tensorboard.writer import SummaryWriter

from sentient_tensorboard.plot import Plot

training_episodes = 0
training_episodes_counter = count(0)


def increment_training_episodes():
    global training_episodes
    training_episodes = next(training_episodes_counter)


class PlotTradingPattern(Plot):
    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        if not self.training and self.episode == 0:  # type:ignore

            self.active = True

            self.mid_prices = deque()
            self.best_buys = deque()
            self.best_sells = deque()
            self.buy_rates = deque()
            self.sell_rates = deque()

    def __call__(self, **kwargs):

        if self.active:

            try:

                best_buy = kwargs["best_buy"]
                best_sell = kwargs["best_sell"]
                buy_rate = kwargs["buy_rate"]
                sell_rate = kwargs["sell_rate"]

                mid_price = (best_buy + best_sell) / 2.0

                self.mid_prices.append(mid_price)
                self.best_buys.append(best_buy)
                self.best_sells.append(best_sell)
                self.buy_rates.append(buy_rate)
                self.sell_rates.append(sell_rate)

                # count = next(self.counter)

                # self.writer.add_scalar("mid_price", mid_price, count)
                # self.writer.add_scalar("best_buy", best_buy, count)
                # self.writer.add_scalar("best_sell", best_sell, count)
                # self.writer.add_scalar("buy_rate", buy_rate, count)
                # self.writer.add_scalar("sell_rate", sell_rate, count)
                # self.writer.add_scalars(
                #     "all",
                #     {
                #         "mid_price": mid_price,
                #         "best_buy": best_buy / mid_price,
                #         "best_sell": best_sell / mid_price,
                #         "buy_rate": buy_rate / mid_price,
                #         "sell_rate": sell_rate / mid_price,
                #     },
                #     count,
                # )
            except KeyError:
                self.active = False
                pass

    def plot(self):

        if self.active:

            xvals = np.arange(len(self.mid_prices))

            # x = np.log10(np.array(self.best_buys) / np.array(self.mid_prices))
            # print(f"{x=}, {len(x)=}")

            # x = np.log10(np.array(self.best_sells) / np.array(self.mid_prices))
            # print(f"{x=}, {len(x)=}")

            # x = np.log10(np.array(self.buy_rates) / np.array(self.mid_prices))
            # print(f"{x=}, {len(x)=}")

            # x = np.log10(np.array(self.sell_rates) / np.array(self.mid_prices))
            # print(f"{x=}, {len(x)=}")

            self.fig = plt.figure(figsize=(18, 10), dpi=80)

            ax = plt.axes()
            ax.set_facecolor("whitesmoke")

            plt.xlabel("Step")
            plt.title(f"Trading Pattern After {training_episodes} Training Episodes")
            plt.ylabel("Log10 of Mid Price Ratio")

            plt.plot(
                xvals,
                np.log10(np.array(self.best_buys) / np.array(self.mid_prices)),
                linewidth=0.5,
                c="g",
            )
            plt.plot(
                xvals,
                np.log10(np.array(self.best_sells) / np.array(self.mid_prices)),
                linewidth=0.5,
                c="r",
            )
            plt.scatter(
                xvals,
                np.log10(np.array(self.buy_rates) / np.array(self.mid_prices)),
                s=5,
                marker="s",  # type:ignore
                c="black",  # type:ignore
            )
            plt.scatter(
                xvals,
                np.log10(np.array(self.sell_rates) / np.array(self.mid_prices)),
                s=5,
                marker="s",  # type:ignore
                c="deepskyblue",
            )

            graph_name = (
                f"trading-pattern-child-process-{self.instance_id}.png"  # type:ignore
            )

            self.writer.add_figure(
                graph_name,
                self.fig,
                global_step=None,
                close=True,
                walltime=None,
            )

            plt.close()

            # plt.savefig(graph_name)


if __name__ == "__main__":

    # Make Artificial Data --------
    nsteps = 1000
    tick = 0.00000001
    fairprice = np.cumsum(np.random.normal(0, 0.00001, nsteps)) + 0.001
    bestbuy = (
        fairprice - tick + np.minimum(0, np.random.normal(-0.000001, 0.000001, nsteps))
    )
    bestsell = (
        fairprice + tick + np.maximum(0, np.random.normal(0.000001, 0.000001, nsteps))
    )
    midprice = (bestbuy + bestsell) / 2

    buyrate = (
        midprice - tick + np.minimum(0, np.random.normal(-0.000001, 0.000001, nsteps))
    )
    sellrate = (
        midprice + tick + np.maximum(0, np.random.normal(0.000001, 0.000001, nsteps))
    )
    # ------------------------------

    plot = PlotTradingPattern(training=False, instance_id=1, episode=0)

    for bestbuy, bestsell, midprice, buyrate, sellrate in zip(
        bestbuy, bestsell, midprice, buyrate, sellrate
    ):
        plot(
            **{
                "best_buy": bestbuy,
                "best_sell": bestsell,
                "buy_rate": buyrate,
                "sell_rate": sellrate,
            }
        )

    plot.plot()
