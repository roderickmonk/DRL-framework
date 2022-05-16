import csv
import logging
import os
import pprint
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
from dataset import Dataset
from orderbook_dataframe_env import ContinuousActionSpaceEnv as env

warnings.filterwarnings("ignore", category=UserWarning)


def rewards_tracer(
    *,
    pdf_filename="sac-test2",
    dataset_filename="gacsell_test_set_snt.json",
    dataset_record=0,
    action=0.2,
    output_csv="rewards_trace.csv",
):

    _log = logging.debug

    pdf_path = "/".join([os.environ.get("CO3_PATH"), "PDFs", pdf_filename])
    pdf = np.genfromtxt(pdf_path, delimiter=",")

    _log(f"{pdf=}")

    env.pdf_x = pdf[:, 0]
    logging.debug(f"{env.pdf_x=}")

    env.pdf_y = pdf[:, 1]
    logging.debug(f"{env.pdf_y=}")

    env.weight = np.sum(env.pdf_y)

    env.ql = 0.2
    env.precision = 8
    env.is_buy = False
    env.tick = 10 ** -env.precision

    dataset = "/".join([os.environ.get("CO3_PATH"), "datasets", dataset_filename])
    df = Dataset().read(source=dataset)

    state = df.iloc[dataset_record][0]
    _log(f"{state=}")

    mid_price = df.iloc[dataset_record][1]
    _log(f"{mid_price=}")

    adjusted_tick = env.tick / mid_price

    result = env.get_reward(
        action=action, state=state, mid_price=mid_price, detailed=True
    )
    _log(pprint.pformat(result))

    with open(output_csv, "w") as f:
        writer = csv.DictWriter(f, fieldnames=result.keys())
        writer.writeheader()
        writer.writerow(result)


if __name__ == "__main__":

    rewards_tracer()
    logging.warning("✨That's✨All✨Folks✨")
