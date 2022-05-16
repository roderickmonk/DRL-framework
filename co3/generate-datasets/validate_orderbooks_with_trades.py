#!/usr/bin/env python
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
import logging
import traceback

from devtools import debug
from env_modules.validate_orderbooks_with_trades import ValidateOrderbooksWithTrades
from omegaconf.errors import ConfigAttributeError
from sentient_util import cfg, logger
from sentient_util.exceptions import Anomaly, InvalidConfiguration

from co3_init import co3_init
from pydantic_config import *


def main() -> None:

    config = co3_init()

    # The config_class defines the pydantic class
    try:
        config = eval(config.agent.config_class)(**config)
    except ConfigAttributeError:
        raise InvalidConfiguration()

    debug(config)

    ValidateOrderbooksWithTrades(env_config=config.env_config)()


if __name__ == "__main__":

    try:
        main()

    except (Anomaly, KeyboardInterrupt, InvalidConfiguration, SystemExit) as msg:
        traceback.print_tb(msg.__traceback__)
        logging.fatal(msg)

    except EOFError as msg:
        # traceback.print_tb(msg.__traceback__)
        logging.fatal(msg)

    else:
        logging.warning("✨That's✨All✨Folks✨")
