#!/usr/bin/env python
import warnings

import gym

warnings.filterwarnings("ignore", category=UserWarning)
import logging
import traceback
from typing import Any, List, Optional

import omegaconf
import torch
from devtools import debug
from devtools import debug as db
from playback import Playback
from sentient_util.exceptions import Anomaly, InvalidConfiguration
from wrappers import ActionWrapper

from co3_init import co3_init
from pydantic_config import *

torch.set_default_dtype(torch.float32)

try:
    profile  # type: ignore
except NameError:
    profile = lambda x: x

from datetime import datetime


@profile  # type:ignore
def main() -> None:

    config = co3_init()

    # The config_class defines the pydantic class
    try:
        config = eval(config.agent.config_class)(**config)
    except omegaconf.errors.ConfigAttributeError:
        config = DefaultProcessConfig(**config)

    debug(config)
    
    env = gym.make(
                config.env_config.env_name, env_config=config.env_config
            )
    env = ActionWrapper(env)

    with Playback(config=config, env=env) as playback:
        logging.warning(f"Last Child Process Mean Reward: {playback()}")


if __name__ == "__main__":

    try:
        main()

    except KeyboardInterrupt:
        logging.fatal("KeyboardInterrupt")

    except (Anomaly, InvalidConfiguration) as msg:
        traceback.print_tb(msg.__traceback__)
        logging.fatal(msg)

    except EOFError as msg:
        traceback.print_tb(msg.__traceback__)
        logging.fatal(msg)

    else:
        logging.warning("✨That's✨All✨Folks✨")

