#!/usr/bin/env python
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import logging

from sentient_util import cfg
import constants
from sentient_util import logger
import torch
import util
from sentient_util.constants import *
from sentient_util.exceptions import InvalidConfiguration
from omegaconf import DictConfig, OmegaConf


def co3_init() -> DictConfig:

    logger.setup()
    logging.warning(f"Python version: {PYTHON_VERSION}")
    logging.warning(f"PyTorch version: {torch.__version__}")
    logging.warning(f"cuda.device_count: {torch.cuda.device_count()}")  # type: ignore

    config = util.load_config(
        default_file="co3.yaml", options="playbacks", selector="playback",
    )

    # Default console is progress bar
    constants.CONSOLE = config.console if config.console else "progress_bar"

    if "agent" not in config:
        raise InvalidConfiguration(f"config.agent missing")

    if "env_config" not in config:
        raise InvalidConfiguration(f"config.env_config missing")

    if "order_depths" in config.env_config:
        cfg.load_order_depths(config.env_config)

    if "time_breaks" in config.env_config:
        cfg.load_time_breaks(config.env_config)

    logging.warning("Playback Configuration")
    # logging.warning(OmegaConf.to_yaml(config))
    # logging.warning(cfg.ConfigObj(config))

    return config

