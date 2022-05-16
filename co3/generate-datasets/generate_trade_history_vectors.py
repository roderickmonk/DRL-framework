import json
import logging
import os

import gym
import util
import numpy as np

# from balance_trade_history_env import BalanceTradeHistoryEnv
from trade_history import TradeHistory

from generate_dataset import GenerateDataset


class GenerateTradeHistoryVectors(GenerateDataset):
    """"""

    def __init__(self, generate_config):

        super().__init__(generate_config)

        self.vector_name = "th_vector"

        self.dataset_data = {"index": [], self.vector_name: [], "it": []}


class GenerateTradeHistoryVectors_Evaluate(GenerateTradeHistoryVectors):
    """"""

    def __init__(self, generate_config):

        self.generate_config = generate_config

        self.env = gym.make(generate_config.env, env_config=generate_config.env_config)

        super().__init__(generate_config)

    def load_row(self, *, index, vector, it):
        """"""

        self.dataset_data["index"].append(int(index))
        self.dataset_data[self.vector_name].append(vector)
        self.dataset_data["it"].append(it)

        counter = next(self._counter)

        if counter % self.generate_config.log_interval == 0:
            logging.info(
                f"{counter=}, {index=}, {self.vector_name}:\n{np.array (vector)} {util.log_interval()}"
            )

    def __call__(self):

        vps = TradeHistory.getInstance().vps

        last_index = 0

        _next_tick = self.env._tick()  # type:ignore
        _next_vector = self.env.generator()  # type:ignore
        self.env.reset()

        while True:

            try:

                index = self.env.current_step  # type:ignore
                next(_next_tick)
                vector = next(_next_vector)

                if index in vps and last_index != index:

                    self.load_row(index=index, vector=vector.tolist(), it=vps[index])
                    last_index = index

            except StopIteration as msg:
                logging.error(msg)
                break

            except KeyboardInterrupt:
                logging.error("Keyboard Interrupt")
                break

        self.save2json()

        self.display_dataset()


"""
class GenerateTradeHistoryVectors_Balance(GenerateTradeHistoryVectors):
    """ """

    def __init__(self, *args, **kwargs):

        self.env = gym.make("sentient_gym:BalanceTradeHistory-v0")
        self.env._configure_environment(*args, **kwargs)

        super().__init__(*args, **kwargs)

    def __call__(self):

        bid = TradeHistory.getInstance().bid
        ask = TradeHistory.getInstance().ask

        last_index = 0

        while True:

            try:

                next(self._tick)
                index = self.env.current_step
                th_vector = next(self.env.generator()).tolist()

                # print(f"{datetime.datetime.fromtimestamp(index)=}")

                if (index in bid or index in ask) and last_index != index:

                    self.load_row(
                        index,
                        th_vector,
                        (
                            bid[index] if index in bid else 0,
                            ask[index] if index in ask else 0,
                        ),
                    )

                    last_index = index

            except (StopIteration, RuntimeError):
                break

        self.save2json(
            file=self.set_output_dataset(
                destination=self.config.generate.th_vectors.destination,
            ),
            dataset_data=self.dataset_data,
        )

        self.display_dataset()
"""
