#!/usr/bin/env python
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
import logging
import os
import traceback

import util
from generate_orderbook_vectors import GenerateOrderbookVectors
from generate_trade_history_vectors import GenerateTradeHistoryVectors_Evaluate
from omegaconf import DictConfig, ListConfig, OmegaConf
from sentient_util import cfg, logger
from sentient_util.constants import PYTHON_VERSION
from sentient_util.exceptions import Anomaly, InvalidConfiguration


def main(config: DictConfig | ListConfig) -> None:

    logger.setup()
    logger.set_log_level("INFO")

    logging.debug(OmegaConf.to_yaml(config))
    logging.warning(f"Python version: {PYTHON_VERSION}")

    os.chdir(os.environ.get("CO3_PATH"))  # type: ignore

    util.display_configuration(cfg.ConfigObj(config), "Generate Configuration")

    if config.env == "sentient_gym:OrderbookHistoryEnv-v0":
        GenerateOrderbookVectors(config)()

    elif config.env == "sentient_gym:EvaluateTradeHistoryEnv-v0":
        GenerateTradeHistoryVectors_Evaluate(config)()

    else:
        raise ValueError(f"Unknown Environment {config.env}")


if __name__ == "__main__":

    try:
        main(
            util.load_config(
                default_file="generate.yaml",
                options="generators",
                selector="generator",
            )
        )

    except KeyboardInterrupt:
        logging.fatal("KeyboardInterrupt")

    except InvalidConfiguration as err:
        logging.fatal(err)

    except Anomaly as err:
        logging.fatal(err)

    else:
        logging.warning("✨That's✨All✨Folks✨")
