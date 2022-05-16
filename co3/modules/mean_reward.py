import os
from datetime import datetime as dt
import logging
from constants import CO3_PATH
import itertools as it

_log = logging.debug


class MeanReward:
    """"""

    _filename = None

    def __init__(self, config):

        self.MR = 0.0  # mean reward
        self.instance_id = config.instance_id

        self._calc_MR_counter = it.count(1)

        if cls._filename is None:

            path = CO3_PATH + "MR"
            os.makedirs(path, mode=0o755, exist_ok=True)

            cls._filename = "/".join(
                [path, dt.now().isoformat().split(".")[0] + ".csv"]
            )

            with open(cls._filename, "a") as f:
                f.write(",".join(["Timestamp", "Process Instance", "MR"]) + "\n")

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):

        self.write()

        msg_prologue = f"Process Instance {self.instance_id}"

        logging.warning(f"{msg_prologue} Complete, Mean Reward: {self.MR:10.6f}")

        if type or value or traceback:
            _log(f"MeanReward __exit__ traceback: {type=}, {value=}, {traceback=}")

    def __call__(self, reward=None):

        if reward is not None:

            if 1 == (calc_MR_counter := next(self._calc_MR_counter)):
                self.MR = reward
            else:
                self.MR = ((calc_MR_counter - 1) * self.MR + reward) / calc_MR_counter

        return self.MR

    def write(self):

        with open(cls._filename, "a") as f:  # type:ignore
            f.write(
                ",".join(
                    [
                        dt.now().isoformat().split(".")[0],
                        str(self.instance_id),
                        str(self.MR),
                    ]
                )
                + "\n"
            )


cls = MeanReward
