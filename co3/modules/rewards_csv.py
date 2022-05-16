from abc import abstractmethod
import logging
import os
from datetime import datetime as dt

from sentient_util.exceptions import Anomaly
from constants import (
    BALANCE_CSV_COLUMNS,
    CO3_PATH,
    CONTINUOUS_ACTION_COLUMNS,
    SIMULATION_ACTION_COLUMNS,
    EVALUATE_CSV_COLUMNS,
)
import itertools as it

_log = logging.debug


class RewardsCsv:
    """"""

    def __init__(self, config):

        self.config = config
        self._file_object = None
        self._iter_count = it.count(0)

        # if not bool(os.environ.get("PYTEST_CURRENT_TEST")) and config.misc.generate_csv:
        if config.misc.generate_csv:

            rewards_path = None

            csv_file_name = (
                dt.now().isoformat().split(".")[0]
                + "-"
                + str(config.instance_id)
                + ".csv"
            )

            if bool(config.misc.csv_path) is False:
                rewards_path = CO3_PATH + "rewards"
            else:

                if os.path.isabs(config.misc.csv_path):
                    rewards_path = config.misc.csv_path

                else:

                    csv_path = config.misc.csv_path

                    if csv_path.split("/")[0] != "rewards":
                        raise ValueError(f"Illegal csv_path: {csv_path}")

                    if csv_path[-1] == "/":
                        csv_path = csv_path[0:-1]

                    if csv_path[0] == "~":
                        rewards_path = csv_path
                    else:

                        if csv_path[-1] == "/":
                            csv_path = csv_path[0:-1]

                        rewards_path = CO3_PATH + csv_path

            if not bool(rewards_path):
                raise Anomaly("Software Anomaly")

            os.makedirs(rewards_path, mode=0o755, exist_ok=True)
            full_path_file_name = "/".join([rewards_path, csv_file_name])

            self._file_object = open(full_path_file_name, "a")

            _log(f"Rewards csv file: {full_path_file_name}")

    @abstractmethod
    def __call__(self):
        raise NotImplementedError

    def close(self):
        if self._file_object is not None:
            _log(f"Close Rewards CSV")
            self._file_object.close()

    @abstractmethod
    def write(self):
        raise NotImplementedError


class EvaluateRewardsCsv(RewardsCsv):
    """"""

    def __init__(self, *, config):

        super().__init__(config=config)
        setattr(config, "rewards_csv", self)

    def __call__(
        self,
        *,
        episode: int,
        step: int,
        reward: float,
        state,
        it: float = None,
        action: int,
        fill_size: float = None,
        pf=None,
        order_depth=None,
    ):
        if self._file_object is not None:

            if next(self._iter_count) == 0 and self._file_object is not None:
                self._file_object.write(",".join(EVALUATE_CSV_COLUMNS) + "\n")

            data = [
                episode,
                step,
                reward,
                [state] if self.config.misc.record_state else "",
                it,
                order_depth,
                pf,
                fill_size,
            ]

            self._file_object.write(
                ",".join([str(x) for x in data if x is not None]) + "\n"
            )


class BalanceRewardsCsv(RewardsCsv):
    """"""

    def __init__(self, *, config):

        super().__init__(config=config)
        setattr(config, "rewards_csv", self)

    def __call__(
        self,
        *,
        episode: int,
        step: int,
        reward: float,
        state,
        bid_it,
        ask_it,
        bid_order_depth,
        ask_order_depth,
        bid_pf,
        ask_pf,
        bid_fill_size: float,
        ask_fill_size: float,
        balance: float,
    ):

        config = self.config

        if self._file_object is not None:

            self._file_object.write(",".join(BALANCE_CSV_COLUMNS) + "\n")

        if self.config.generate_csv and self._file_object is not None:

            bid_it, ask_it = self.env.get_it().values()  # type:ignore

            data = [
                episode,
                step,
                reward,
                [state] if self.config.record_state else "",
                bid_it,
                ask_it,
                bid_order_depth,
                ask_order_depth,
                bid_pf,
                ask_pf,
                bid_fill_size,
                ask_fill_size,
                balance,
            ]

            self._file_object.write(
                ",".join([str(x) for x in data if x is not None]) + "\n"
            )


class ContinuousRewardsCsv(RewardsCsv):
    """"""

    def __init__(self, *, config):
        super().__init__(config=config)
        self.config = config

    def __call__(
        self, *, episode, step, reward, state, action, expected_profit, predicted
    ):

        if self._file_object is not None:

            if next(self._iter_count) == 0:
                self._file_object.write(",".join(CONTINUOUS_ACTION_COLUMNS) + "\n")

            data = [
                episode,
                step,
                reward,
                [state] if self.config.misc.record_state else "",
                action,
                expected_profit,
                predicted,
            ]

            self._file_object.write(
                ",".join([str(x) for x in data if x is not None]) + "\n"
            )


class SimulationRewardsCsv(RewardsCsv):
    """"""

    def __init__(self, *, config):
        super().__init__(config=config)
        self.config = config

    def __call__(self, **kwargs):

        if self._file_object is not None:

            if next(self._iter_count) == 0:
                self._file_object.write(",".join(SIMULATION_ACTION_COLUMNS) + "\n")

            data = [
                kwargs["ts"],
                kwargs["episode"],
                kwargs["step"],
                f"{kwargs['buy_rate']:.8f}",
                f"{kwargs['sell_rate']:.8f}",
                f"{kwargs['reward']:.8f}",
                f"{kwargs['delta_funds']:.8f}",
                f"{kwargs['delta_inventory']:.8f}",
                f"{kwargs['best_buy']:.8f}",
                f"{kwargs['best_sell']:.8f}",
                kwargs["buy_match"].value,
                kwargs["buy_Rs"],
                kwargs["buy_Qs"],
                kwargs["sell_match"].value,
                kwargs["sell_Rs"],
                kwargs["sell_Qs"],
                f"{kwargs['funds']:.8f}",
                f"{kwargs['inventory']:.8f}",
                f"{kwargs['balance']:.8f}",
            ]

            from devtools import debug

            if kwargs["reward"] != 0.0:
                pass  # debug(data)

            self._file_object.write(
                ",".join([str(x) for x in data if x is not None]) + "\n"
            )
