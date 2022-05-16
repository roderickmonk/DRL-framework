#!/usr/bin/env python
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import logging

import optuna
from sentient_util.exceptions import Anomaly, InvalidConfiguration
from optuna.trial import TrialState
from playback import Playback
import itertools as it
from co3_init import co3_init

try:
    profile  # type: ignore
except NameError:
    profile = lambda x: x


step = it.count(0)


class Objective(object):
    def __init__(self, config):
        self.config = config

    def __call__(self, trial: optuna.Trial) -> float:

        config.agent.agent = trial.suggest_categorical(
            "agent", self.config.agent.categorical.agent
        )

        config.agent.gamma = trial.suggest_float(
            "gamma", *self.config.agent.float_interval.gamma, log=False
        )

        config.agent.actor_lr = trial.suggest_float(
            "actor_lr", *self.config.agent.float_interval.actor_lr, log=False
        )

        config.agent.critic_lr = trial.suggest_float(
            "critic_lr", *self.config.agent.float_interval.critic_lr, log=False
        )

        config.agent.tau = trial.suggest_float(
            "tau", *self.config.agent.float_interval.tau, log=False
        )

        config.agent.training_interval = trial.suggest_int(
            "training_interval", *self.config.agent.int_interval.training_interval,
        )

        config.agent.exploration = trial.suggest_int(
            "exploration", *self.config.agent.int_interval.exploration,
        )

        print(f"{config.agent.gamma=}")

        with Playback(config=config) as playback:
            MR = playback()
            logging.warning(f"Last Child Process Mean Reward: {MR}")

            trial.report(MR, next(step))

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            return MR


if __name__ == "__main__":

    try:
        config = co3_init()

        study = optuna.create_study()
        study.optimize(Objective(config), n_trials=1)

        pruned_trials = study.get_trials(
            deepcopy=False, states=[TrialState.PRUNED]  # type:ignore
        )
        complete_trials = study.get_trials(
            deepcopy=False, states=[TrialState.COMPLETE]  # type:ignore
        )

        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

    except KeyboardInterrupt:
        logging.fatal("KeyboardInterrupt")

    except InvalidConfiguration as err:
        logging.fatal(err)

    except Anomaly as err:
        logging.fatal(err)

    else:
        logging.warning("✨That's✨All✨Folks✨")

