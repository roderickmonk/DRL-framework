import os
from collections import namedtuple
from datetime import date

import numpy as np
import torch
from torch import cuda
from torch.backends import cudnn

METADATA_COLLECTION = "V3-recovered-orderbooks-metadata"

noop = lambda: None

Experience = namedtuple(
    "Experience", field_names=["state", "action", "next_state", "reward", "done"],
)

if cuda.is_available():
    cudnn.enabled = False


CONSOLE = "logging"

CONTINUOUS_ACTION_ENVIRONMENTS = [
    "sentient_gym:BuyOrderbookDataFrameEnv-v0",
    "sentient_gym:SellOrderbookDataFrameEnv-v0",
]

DISCRETE_ACTION_ENVIRONMENTS = [
    "sentient_gym:EvaluateDataFrameEnv-v0",
    "sentient_gym:BalanceDataFrameEnv-v0",
]

CONTINUOUS_ACTION_COLUMNS = [
    "Episode",
    "Step",
    "Reward",
    "State",
    "Action",
    "Expected Profit",
    "Predicted",
]

SIMULATION_ACTION_COLUMNS = [
    "ts",
    "Episode",
    "Step",
    "BuyRate",
    "SellRate",
    "Reward",
    "\u0394" + "Funds",
    "\u0394" + "Inventory",
    "BuyTop",
    "SellTop",
    "BuyMatch",
    "BuyRs",
    "BuyQs",
    "SellMatch",
    "SellRs",
    "SellQs",
]

EVALUATE_CSV_COLUMNS = [
    "Episode",
    "Step",
    "Reward",
    "State",
    "Imminent Trade",
    "Order Depth",
    "Profit Factor",
    "Fill Size",
]

BALANCE_CSV_COLUMNS = [
    "Episode",
    "Step",
    "Reward",
    "State",
    "Imminent Buy",
    "Imminent Sell",
    "Order Depth Buy",
    "Order Depth Sell",
    "Profit Factor Buy",
    "Profit Factor Sell",
    "Buy Fill Size",
    "Sell Fill Size",
    "Balance",
]


DEVICE = torch.device("cuda:0" if cuda.is_available() else "cpu")


# OPTIMIZED = bool(sys.flags.optimize)

# # EPS is the smallest representable positive number
# # such that 1.0 + EPS != 1.0.
# EPS = np.finfo(np.float32).eps.item()

CO3_PATH = os.environ.get("CO3_PATH") + "/"  # type:ignore

# BASENAME = os.path.splitext(os.path.basename(sys.argv[0]))[0]
# LOG_PATH = f"{CO3_PATH}logs/{date.today().isoformat()}"

