import itertools
import logging
import os
from datetime import datetime as dt

from sentient_util.exceptions import Anomaly

from constants import CO3_PATH

try:
    profile  # type:ignore
except NameError:
    profile = lambda x: x


class NetworkTrace:
    """"""

    instance = itertools.count(0)

    pattern = None

    def __init__(self, config):

        self._file_object = None

        if (
            hasattr(config, "nn_trace")
            and not config.training
            and hasattr(config.nn_trace, "active")
            and config.nn_trace.active
        ):

            instance = next(NetworkTrace.instance)

            if not hasattr(config.nn_trace, "count"):
                setattr(config.nn_trace, "count", 3)

            if NetworkTrace.pattern is None:
                if not hasattr(config.nn_trace, "pattern"):
                    NetworkTrace.pattern = [0, 1]
                else:
                    NetworkTrace.pattern = config.nn_trace.pattern

            if len(NetworkTrace.pattern) != 2:
                ValueError(f"nn_trace.pattern {NetworkTrace.pattern} Malformed")

            # Decide if tracing required
            if (
                instance in range(NetworkTrace.pattern[0])
                or (instance - NetworkTrace.pattern[0] + 1) % NetworkTrace.pattern[1]
                == 0
            ):

                logging.debug(f"{instance=}")

                out_directory = None

                file_name = (
                    dt.now().isoformat().split(".")[0]
                    + "-"
                    + str(config.instance_id)
                    + ".csv"
                )

                if (
                    not hasattr(config.nn_trace, "target_directory")
                    or config.nn_trace.target_directory is None
                ):
                    out_directory = CO3_PATH + "nn_traces"
                elif os.path.isabs(config.nn_trace.target_directory):
                    out_directory = config.nn_trace.path
                else:

                    path = config.nn_trace.target_directory

                    if path[0:1] == "./":
                        path = path[2:]

                    if path[-1] == "/":
                        path = path[0:-1]

                    if path[0] == "~":
                        out_directory = path
                    else:
                        out_directory = CO3_PATH + "nn_traces/" + path

                if not bool(out_directory):
                    raise Anomaly("Software Anomaly")

                os.makedirs(out_directory, mode=0o755, exist_ok=True)
                nn_trace_file = "/".join([out_directory, file_name,])
                logging.debug(f"nn trace file: {nn_trace_file}")

                self._file_object = open(nn_trace_file, "a")
                self._file_object.write(
                    ",".join(["Episode", "State", "Output Layer"]) + "\n"
                )
                setattr(config.nn_trace, "network_trace", self)

            else:
                delattr(config.nn_trace, "network_trace")

    def close(self):
        if bool(self._file_object):
            logging.debug(f"Close network trace")
            if hasattr(self._file_object, "close"):
                self._file_object.close()  # type:ignore

    def write(self, data):
        self._file_object.write(  # type:ignore
            ",\n".join([str(x) for x in data if x is not None]) + "\n"
        )
