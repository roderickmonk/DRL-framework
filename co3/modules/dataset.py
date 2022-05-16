import json
import os

import numpy as np
import pandas as pd
from constants import CO3_PATH
from util import fext
from pandas import DataFrame


def read(*, source: str):

    try:
        with open(fext(source, ".json"), "r") as fp:
            dataset = json.load(fp)
    except json.decoder.JSONDecodeError as msg:
        raise RuntimeError(f"Corrupt json file: {source}, reason: {msg}")

    index = dataset.pop("index", None)

    df = DataFrame(dataset, index=index, columns=dataset.keys())

    return df


class Dataset:

    dataset_data = {"index": [], "state": [], "it": []}
    dataset_file = None

    @staticmethod
    def save():

        # Remove any previous version
        try:
            if Dataset.dataset_file:
                os.remove(fext(Dataset.dataset_file, "json"))
        except FileNotFoundError:
            pass

        json_data = json.dumps(Dataset.dataset_data)

        if Dataset.dataset_file:
            with open(fext(Dataset.dataset_file, "json"), "w") as fp:
                fp.write(json_data)
                fp.close()

    def __init__(self, _=None):

        self.datasets_path = CO3_PATH + "datasets"

    def read(self, *, source: str) -> DataFrame:
        """Loads a file into a pandas DataFrame"""

        try:

            if source in self.ls():

                with open(source, "r") as fp:
                    try:
                        dataset = json.load(fp)
                    except KeyboardInterrupt:
                        raise

                index = dataset.pop("index", None)

                df = pd.DataFrame(dataset, index=index, columns=dataset.keys())

                return df

            else:
                raise RuntimeError(f"{source=} not found")

        except KeyboardInterrupt:
            raise

    ls = lambda self: [
        "/".join([x[0], y]) for x in os.walk(self.datasets_path) for y in x[2]
    ]

    def purge(self, *, dataset: str):
        # Delete the dataset from the list of datasets

        try:
            os.remove(dataset)
        except FileNotFoundError:
            pass

    def persist(self, *, source: DataFrame, dest: str, overwrite: bool = False) -> None:
        """Save a DataFrame to file"""

        assert overwrite or not os.path.exists(dest), f"Data-set {dest=} already exists"

        self.purge(dataset=dest)

        dataset_data = {"index": []}

        for index, row in source.iterrows():

            dataset_data["index"].append(index)

            # Peel the data from the DF column by column
            for column in source.columns:

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

        # Save to a file
        json_data = json.dumps(dataset_data)
        with open(fext(dest, "json"), "w") as fp:
            fp.write(json_data)

    def _dataset_file(self, *, dataset):

        # Define the location of the dataset
        dataset_path = CO3_PATH + "datasets"
        os.makedirs(dataset_path, mode=0o755, exist_ok=True)

        return "/".join([dataset_path, dataset])

