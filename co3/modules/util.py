import logging
import math
import os
import sys
import time
from datetime import date, datetime, timedelta
from typing import List

import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from sentient_util.constants import BASENAME, IS_CRON_TASK, LOG_PATH
from sentient_util.exceptions import Anomaly
from dateutil.parser import parse
from matplotlib import pyplot as plt
from omegaconf import DictConfig, ListConfig, OmegaConf
from pymongo import ASCENDING, MongoClient
from pymongo.collection import Collection
from sortedcontainers import SortedList
from torch import nn
from tqdm import tqdm

from constants import CO3_PATH, METADATA_COLLECTION

log_interval_start = None


def log_interval():
    global log_interval_start
    ret = "" if log_interval_start is None else f"+{time.time()-log_interval_start:.3}"
    log_interval_start = time.time()
    return ret


def display_configuration(config, label):

    logging.warning(f"+++ {label}")
    logging.warning(f"{config}")
    logging.warning(f"+++ End Configuration")


def fext(filename: str, ext: str) -> str:

    ext = ext if ext[0] == "." else "." + ext
    assert len(ext) > 1

    f_name, f_ext = os.path.splitext(filename)

    if not (len(f_ext) == 0 or f_ext == ext):
        raise ValueError(f"{filename} has an inappropriate file extension {f_ext=}")

    if os.path.exists(filename):
        return filename
    elif os.path.exists(f_name):
        return f_name
    else:
        return f_name + ext


def set_path(*, configured_path, nominal_folder, nominal_ext) -> str:
    """"""

    if configured_path is None:

        folder = CO3_PATH + nominal_folder
        os.makedirs(folder, mode=0o755, exist_ok=True)

        return "/".join([folder, datetime.now().isoformat().split(".")[0] + ".pt"])

    elif os.path.isabs(configured_path):

        folder, _ = os.path.split(configured_path)
        os.makedirs(folder, mode=0o755, exist_ok=True)

        return fext(configured_path, ".pt")

    elif not os.path.isabs(configured_path):

        folder, _ = os.path.split(configured_path)

        if folder.split("/")[0] != nominal_folder:
            raise ValueError(f"Illegal path: {configured_path}")

        os.makedirs(CO3_PATH + folder, mode=0o755, exist_ok=True)
        return fext(CO3_PATH + configured_path, nominal_ext)

    raise Anomaly("path unbound")


def create_log_gaussian(mean, log_std, t):
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p


def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs


def apply_depth(depth: float, orderbook: np.ndarray) -> np.ndarray:

    return orderbook[
        0 : 1
        + np.searchsorted(
            np.cumsum(np.prod(orderbook, axis=1)),
            np.inf if depth == 0 else depth,
            side="left",
        )
    ]


def load_config(*, default_file: str, options: str, selector: str) -> DictConfig:
    """"""

    def get_config_path(config_file_name) -> str:

        if (co3_config_path := os.environ.get("CO3_CONFIG_PATH")) :
            for path in co3_config_path.split(":"):
                for file in os.listdir(path):
                    if file == config_file_name:
                        return "/".join([path, file])

        raise FileNotFoundError(f"{config_file_name}")

    config_file_name = default_file

    for i, param in enumerate(sys.argv):
        param = param.split("=")
        if len(param) == 2:
            key, value = param
            if key == "config":
                config_file_name = fext(value, "yaml")
                del sys.argv[i]  # config= is not used by omegaconf

    config = DictConfig(OmegaConf.load(get_config_path(config_file_name)))

    # Must have at least one option
    if options not in config or len(option_keys := config[options].keys()) == 0:
        raise ValueError(f"{config_file_name} Invalid")

    # If only one option, it is implicitly selected
    if len(option_keys) == 1:
        config[selector] = list(option_keys)[0]

    defaults = (
        OmegaConf.load(config.defaults_)
        if "defaults_" in config
        else OmegaConf.create()  # empty object
    )

    # Merge with the cli
    config = DictConfig(OmegaConf.merge(config, OmegaConf.from_cli()))

    # console is a special case
    console = OmegaConf.from_cli().get("console", config.console)

    config = OmegaConf.merge(
        defaults, config[options][config[selector]], {"console": console}
    )

    return DictConfig(config)


def tools_configuration(*, window: int = 7, custom: dict = {}) -> DictConfig:

    default_end_range = date.today()
    default_start_range = default_end_range - timedelta(days=window)

    return DictConfig(
        OmegaConf.merge(
            {
                "envId": 0,
                "exchange": "bittrex",
                "start_range": default_start_range.isoformat(),
                "end_range": default_end_range.isoformat(),
                "markets": None,
                "bin_size": 600,  # seconds
            },
            custom,
            OmegaConf.from_cli(),
        )
    )


def get_all_markets(
    *,
    collection=MongoClient(host=os.environ["MONGODB"])["derived-history"][
        METADATA_COLLECTION
    ],
    start_range,
    end_range,
    envId,
    exchange,
) -> List[str]:

    return list(
        collection.distinct(
            "m",
            {
                "$and": [
                    {"e": envId},
                    {"x": exchange},
                    {"ts": {"$gte": start_range}},
                    {"ts": {"$lt": end_range}},
                ]
            },
        )
    )


def sort_markets(*, config, window=1, markets: List[str]) -> List[str]:

    sorted_list = SortedList()

    for market in markets:

        filter = {
            "$and": [
                {"e": config.envId},
                {"x": config.exchange},
                {"m": market},
                {"day": {"$gte": datetime.now() - timedelta(days=window)}},
            ]
        }

        derived_history = MongoClient(host=os.environ["MONGODB"])["derived-history"]

        orderbooks_heatmap_collection = derived_history["orderbooks_heatmap"]

        histograms = [
            x["histogram"] for x in list(orderbooks_heatmap_collection.find(filter))
        ]

        if len(histograms) == 0:
            # If no histograms exist for the market, then the market
            # is placed at the bottom of the heatmap
            sorted_list.add([0, market])

        else:
            sorted_list.add([np.mean(np.concatenate(*[np.array(histograms)])), market],)

        logging.debug(f"{[*reversed(sorted_list)]=}")

    return [x[1] for x in reversed(sorted_list)]


def test_df(*, states, actions, labels, net):
    """"""

    def de_normalize(data):
        return 10 ** (data) - 1e-12

    df = pd.DataFrame(columns=["Action", "Predicted", "Target", "State_Sum"])

    logging.debug(f"Testing {states.shape=}")
    logging.debug(f"Testing {actions.shape=}")

    with torch.no_grad():

        output = net(states, actions)

        print(f"Testing MAE: {torch.mean(torch.abs(output - labels))}")

        for i in range(len(actions)):
            df = df.append(
                {
                    "Action": f"{actions[i].item():.7f}",
                    "Predicted": f"{output[i].item():.7f}",
                    "Target": f"{labels[i].item():.7f}",
                    "State_Sum": f"{torch.sum(de_normalize(states[i])).item():.7f}",
                },
                ignore_index=True,
            )
    return df


