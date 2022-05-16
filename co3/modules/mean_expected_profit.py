import os
from datetime import datetime as dt
import logging
from constants import CO3_PATH
import itertools as it

_log = logging.debug


class MeanExpectedProfit:
    """"""

    def __init__(self):

        self.value = 0.0

        self.counter = it.count(1)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        pass

    def __call__(self, new_value=None):

        if new_value is not None:

            if 1 == (counter := next(self.counter)):
                self.value: float = new_value
            else:
                self.value: float = ((counter - 1) * self.value + new_value) / counter

        # print(f"{self.value=}")
        # print(f"{type(self.value)=}")

        try:
            return self.value.item()  # type:ignore
        except AttributeError:
            return self.value

