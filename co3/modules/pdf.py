import logging
import os
import warnings
from datetime import datetime, timedelta

import dateutil
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymongo

from constants import CO3_PATH

warnings.filterwarnings("ignore", category=UserWarning)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymongo
from omegaconf import DictConfig, OmegaConf

np.set_printoptions(precision=10, floatmode="fixed", edgeitems=8)  # type:ignore


def get(env_config):

    _log = logging.error

    def pdf_obj():

        return {
            "name": env_config.pdf.name,
            "created": datetime.now(),
            "x": pdf_x.tolist(),
            "y": pdf_y.tolist(),  # type:ignore
        }

    with pymongo.MongoClient(host=os.environ["MONGODB"]) as mongo_client:

        pdf_collection = mongo_client["configuration"]["PDFs"]

        PDF = pdf_collection.find_one({"name": env_config.pdf.name})

        if PDF is None:

            _log(f"Generating New PDF {env_config.pdf.name}")

            pdf_x, pdf_y = _generate(env_config)

            pdf_collection.insert_one(pdf_obj())

        elif "created" not in PDF:

            _log(f"Regenerating PDF {env_config.pdf.name}")

            pdf_x, pdf_y = _generate(env_config)

            pdf_collection.replace_one(
                filter={"_id": PDF["_id"]}, replacement=pdf_obj(), upsert=True,
            )

        else:

            def regenerate():

                while True:

                    regenerate = input("Aged PDF - regenerate? (y/n) ")

                    if regenerate[0].lower() == "y":
                        return True
                    if regenerate[0].lower() == "n":
                        return False

            if (
                datetime.now() - PDF["created"]
                > timedelta(days=env_config.pdf.age_limit)
                and regenerate()
            ):

                logging.critical(f"Regenerating PDF {env_config.pdf.name}")

                pdf_x, pdf_y = _generate(env_config)

                pdf_collection.replace_one(
                    filter={"_id": PDF["_id"]}, replacement=pdf_obj(), upsert=True,
                )

            else:
                # Use the as-retrieved PDF
                pdf_x, pdf_y = np.array(PDF["x"]), np.array(PDF["y"])

        # Save the PDF as a file
        path = CO3_PATH + "PDFs/"
        os.makedirs(path, mode=0o755, exist_ok=True)

        with open(path + env_config.pdf.name, "w") as f:
            for x, y in zip(pdf_x, pdf_y):
                f.write(f"{10**x},{y}\n")

        return 10 ** pdf_x, pdf_y


def _generate(env_config):

    _log = logging.debug

    with pymongo.MongoClient(host=os.environ["MONGODB"]) as mongo_client:

        if env_config.pdf.start_window and isinstance(
            env_config.pdf.start_window, int
        ):
            from_time = datetime.now() - timedelta(days=env_config.pdf.start_window)
            to_time = datetime.now()
        else:
            from_time = dateutil.parser.parse(env_config.pdf.start_range)  # type:ignore
            to_time = dateutil.parser.parse(env_config.pdf.end_range)  # type:ignore

        _log(f"{from_time=}, {to_time=}")

        trades_raw = list(
            mongo_client["history"]["trades"].find(
                {
                    "e": env_config.pdf.envId,
                    "x": env_config.pdf.exchange,
                    "m": env_config.pdf.market,
                    "ts": {"$gte": from_time, "$lt": to_time,},
                },
                no_cursor_timeout=True,
            )
        )

        trades = pd.DataFrame(trades_raw)

        print (f"{trades=}")

        # Need trade volumes in base
        trades["t"] = trades.r * trades.q

        # Remove those of negligible size
        trades = trades[trades.t > 0.0005]

        trades["ts_int"] = trades.ts.astype(np.int64) / 1000000
        trades["seconds"] = np.int64(trades.ts_int / 1000)

        # Get volume per second
        vps = trades.groupby([trades["seconds"]])["t"].sum()

        pdf_y, pdf_x, *_ = np.histogram(
            np.log10(vps.values),  # type:ignore
            bins=env_config.pdf.bins,
        )

        # Center pdf_x within each interval
        pdf_x = 0.5 * (pdf_x[:-1] + pdf_x[1:])  # type:ignore

        for xy in zip(pdf_x, pdf_y):
            logging.debug(xy)

        if env_config.pdf.graph:
            plt.plot(pdf_x, pdf_y, label="PDF")
            plt.show()

        return pdf_x, pdf_y

