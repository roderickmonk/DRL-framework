import json
import logging
import os
import traceback
from pathlib import Path
from typing import Any

import gym
import numpy as np
import pandas as pd
import pdf
import util
from sentient_util.exceptions import Anomaly

from generate_dataset import GenerateDataset


class GenerateOrderbookVectors(GenerateDataset):
    """"""

    def __init__(self, generate_config):
        """"""

        super().__init__(generate_config)  # type:ignore

        self.generate_config = generate_config

        self.env = gym.make(generate_config.env, env_config=generate_config.env_config)

        lookup_vector_name = {
            "buy": "buy_ob_vector",
            "sell": "sell_ob_vector",
            "ob": "ob_vector",
        }

        self.side = generate_config.env_config.side
        self.vector_name = lookup_vector_name[self.side]
        self.dataset_data = {
            "index": [],
            self.vector_name: [],
            "mid_price": [],
        }

        # self.pdf_x, self.pdf_y = pdf.get(self.generate_config.env_config)
        # self.weight = np.sum(self.pdf_y)
        # self.ql = self.generate_config.ql # Not required since targets are no longer included

        self.is_buy = self.generate_config.is_buy

        self.market_tick = (
            10 ** -self.generate_config.precision
            if self.generate_config.is_buy
            else -(10 ** -self.generate_config.precision)
        )
        self.precision = self.generate_config.precision

    def persist(self, df):
        """Save DataFrame"""

        folder, _ = os.path.split(self.generate_config.destination)
        os.makedirs(folder, mode=0o775, exist_ok=True)

        dataset_data = {"index": []}

        for index, row in df.iterrows():

            dataset_data["index"].append(index)

            # Peel the data from the DF column by column
            for column in df.columns:

                if column not in dataset_data:
                    if isinstance(row[column], np.ndarray):
                        dataset_data[column] = [
                            [r.item() for r in row[column].tolist()]
                        ]
                    else:
                        dataset_data[column] = [row[column]]
                else:
                    if isinstance(row[column], np.ndarray):
                        dataset_data[column].append(
                            [r.item() for r in row[column].tolist()]
                        )
                    else:
                        dataset_data[column].append(row[column])

        json_data = json.dumps(dataset_data)
        with open(util.fext(self.generate_config.destination, "json"), "w") as fp:
            fp.write(json_data)

    def load_row(self, *, index, vector, mid_price):
        """"""

        self.dataset_data["index"].append(int(index))
        self.dataset_data[self.vector_name].append(vector)
        self.dataset_data["mid_price"].append(mid_price)

        counter = next(self._counter)
        if counter % self.generate_config.log_interval == 0:
            logging.debug(
                f"\ncounter: {counter},\n{index=}, "
                f"\n{self.vector_name}:\n{np.array (vector)} "
                f"\n{mid_price=} "
                f"{util.log_interval()}"
            )

    def __call__(self):

        while True:

            try:

                mid_price: Any

                (
                    index,
                    # orderbook,  # Not currently being used
                    OB_vector,
                    mid_price,
                    done,
                ) = self.env.step(0)  

                if done:
                    break

                self.load_row(
                    index=index, vector=OB_vector.tolist(), mid_price=mid_price,
                )

            except KeyboardInterrupt:
                logging.error(f"Keyboard Interrupt")
                break

            except BaseException as msg:
                traceback.print_tb(msg.__traceback__)
                logging.error(f"Exception Raised: {msg}")
                break

        self.save_vectors(self.filter_for_uniqueness())

    def filter_for_uniqueness(self):

        index = self.dataset_data.pop("index", None)

        df = pd.DataFrame(self.dataset_data, columns=self.dataset_data.keys(),)  # type: ignore

        unique_vectors, unique = np.unique(df[self.vector_name], return_index=True,)  # type: ignore

        index = pd.DataFrame(np.array(index)[unique], columns=["index"])
        vectors = pd.DataFrame(unique_vectors, columns=[self.vector_name])
        mid_prices = pd.DataFrame(
            np.array(self.dataset_data["mid_price"])[unique], columns=["mid_price"]
        )

        if vectors.shape[0] != index.shape[0] or mid_prices.shape[0] != index.shape[0]:
            raise Anomaly("Software Anomaly")

        # The modified dataframe contains only unique OBs
        df = (
            pd.concat([index, vectors, mid_prices], axis=1)
            .set_index("index")
            .sort_index()
        )
        assert df.shape[1] == 2

        return df

    def save_vectors(self, df):

        self.persist(df)

        logging.info(f"DF with unique orderbook vectors:\n{df}")

        # Do a read back check
        assert df.equals(self.read_json()), "Dataframe reback test failed"

