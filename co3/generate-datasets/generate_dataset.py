import itertools as it
import json
import logging
import os
from typing import Any

import numpy as np
import pandas as pd
from util import fext


class GenerateDataset:
    """"""

    def __init__(self, generate_config: Any):

        self.generate_config = generate_config

        if os.path.isabs(generate_config.destination):
            if generate_config.destination.split("/")[1] != "datasets":
                raise ValueError(
                    f"Illegal datasets path: {generate_config.destination}"
                )

        else:
            if generate_config.destination.split("/")[0] != "datasets":
                raise ValueError(
                    f"Illegal datasets path: {generate_config.destination}"
                )

        # if generate_config.destination.split("/")[0] != "datasets":
        #     raise ValueError(f"Illegal path: {generate_config.destination}")

        self._counter = it.count(0)

    def display_dataset(self):
        logging.info(f"df=\n{self.read_json()}")

    def save2json(self):

        folder, _ = os.path.split(self.generate_config.destination)
        if bool(folder):
            os.makedirs(folder, mode=0o775, exist_ok=True)

        # Remove previous version
        try:
            os.remove(fext(self.generate_config.destination, "json"))
        except FileNotFoundError:
            pass

        json_data = json.dumps(self.dataset_data)  # type:ignore
        with open(fext(self.generate_config.destination, "json"), "w") as fp:
            fp.write(json_data)
            fp.close()

    def read_json(self):

        with open(fext(self.generate_config.destination, "json"), "r") as fp:
            dataset = json.load(fp)

        index = dataset.pop("index", None)

        df = pd.DataFrame(dataset, index=index, columns=dataset.keys())

        if hasattr(df, "state"):
            if not all(
                len(element) == len(df.state.values[0]) for element in df.state.values
            ):
                raise ValueError("Dataset constains inconsistent state sizes")

        if hasattr(df, "ob_vector"):
            if not all(
                len(element) == len(df.ob_vector.values[0])
                for element in df.ob_vector.values
            ):
                raise ValueError("Dataset constains inconsistent ob_vector sizes")

        if hasattr(df, "buy_ob_vector"):
            if not all(
                len(element) == len(df.buy_ob_vector.values[0])
                for element in df.buy_ob_vector.values
            ):
                raise ValueError("Dataset constains inconsistent buy_ob_vector sizes")

        if hasattr(df, "sell_ob_vector"):
            if not all(
                len(element) == len(df.sell_ob_vector.values[0])
                for element in df.sell_ob_vector.values
            ):
                raise ValueError("Dataset constains inconsistent sell_ob_vector sizes")

        if hasattr(df, "th_vector"):
            if not all(
                len(element) == len(df.th_vector.values[0])
                for element in df.th_vector.values
            ):
                raise ValueError("Dataset constains inconsistent th_vector sizes")

        return df

