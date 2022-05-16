import itertools as it
import logging
import math
from contextlib import contextmanager
from time import perf_counter
from typing import Any

import constants
import torch
from constants import DEVICE, Experience
from devtools import debug
from mean_expected_profit import MeanExpectedProfit
from mean_reward import MeanReward
from noise import set_action_noise
from project_functions import *
from pydantic_config import *
from replay_buffer import ReplayBuffer
from sentient_util.constants import OPTIMIZED
from sentient_util.enums import ActionSpace
from sentient_util.matching_engine import MatchResult
from torch import Tensor, nn
from torch.optim import Adam
from tqdm.auto import trange

from agent.agent import AbstractAgent, ProcessConfigs

torch.set_default_dtype(torch.float32)
from sentient_tensorboard.plot_reward import PlotReward
from sentient_tensorboard.plot_trading_pattern import (
    PlotTradingPattern,
    increment_training_episodes,
)


class ContinuousActionSpaceAgent(AbstractAgent):
    """"""

    def __init__(
        self,
        *,
        config: ProcessConfigs,
        env,
        actor_class=None,
        critic_class=None,
    ):

        super().__init__(config=config, env=env)

        if not self.pytest and self.env.action_space_type != ActionSpace.Continuous:
            raise ValueError("Agent not compatible with environment")

        self.noise = set_action_noise(
            action_size=self.env.action_space.shape[0], config=config  # type:ignore
        )

        self.state_size = math.prod(self.env.observation_space.shape)  # type:ignore

        self.action_size = self.env.action_space.shape[0]  # type:ignore

        self.action_space_low = self.env.action_space.low[0]  # type:ignore
        self.action_space_high = self.env.action_space.high[0]  # type:ignore

        self.action_scale = float_tensor(
            (self.env.action_space.high - self.env.action_space.low)  # type:ignore
            / 2.0
        )
        self.action_bias = float_tensor(
            (self.env.action_space.high + self.env.action_space.low)  # type:ignore
            / 2.0
        )

        self.trace_counter = it.count(0)

        self.exploration = config.agent.exploration

        self.pytest_success_threshold = -200

        if actor_class:
            self.actor = actor_class(env, config)
            self.actor_optimizer = Adam(
                self.actor.parameters(), lr=config.agent.actor_lr
            )
            self.actor_target = actor_class(env, config)

        if critic_class:
            self.critic = critic_class(env, config)
            self.critic_optimizer = Adam(
                self.critic.parameters(), lr=config.agent.critic_lr
            )
            self.critic_target = critic_class(env, config)

        self.load()

        super().hard_update(self.actor_target, self.actor)
        super().hard_update(self.critic_target, self.critic)

        self.loss = config.agent.loss

        self.csv_write = None

    def __enter__(self):

        # The following line is required
        from rewards_csv import ContinuousRewardsCsv, SimulationRewardsCsv

        self.rewards_csv = eval(self.config.misc.csv_class)(config=self.config)

        if self.config.misc.csv_class == "ContinuousRewardsCsv":
            setattr(self, "csv_write", self.profit_csv_write)
        elif self.config.misc.csv_class == "SimulationRewardsCsv":
            setattr(self, "csv_write", self.sim_csv_write)
        else:
            raise RuntimeError("No CSV Class defined")

        return self

    def __exit__(self, *args):
        self.rewards_csv.close()

    def select_action(self, state: Tensor) -> Tensor:

        with torch.no_grad():

            action = self.actor(state.view(1, -1)) + self.noise()

            return torch.clamp(
                action.squeeze(0), self.action_space_low, self.action_space_high
            )

    def profit_csv_write(self, **kwargs):

        try:
            expected_profit = (
                kwargs["step_info"]["expected_profit"] if not self.pytest else 0
            )
        except KeyError:
            expected_profit = 0.0

        self.rewards_csv(
            episode=kwargs["episode"],
            step=kwargs["step"],
            reward=kwargs["reward"].item(),
            action=kwargs["action"],
            state=kwargs["state"],
            expected_profit=expected_profit,
            predicted=kwargs["predicted"],
        )

    def sim_csv_write(self, **kwargs):

        buy_match = MatchResult.NO_TRADES
        buy_Rs = ""
        buy_Qs = ""
        sell_match = MatchResult.NO_TRADES
        sell_Rs = ""
        sell_Qs = ""

        if kwargs["step_info"]["matching"]:
            (
                buy_match,
                buy_trading_activity,
                sell_match,
                sell_trading_activity,
            ) = kwargs["step_info"]["matching"]
            buy_Rs = buy_trading_activity.trade_Rs
            buy_Qs = buy_trading_activity.trade_Qs
            sell_Rs = sell_trading_activity.trade_Rs
            sell_Qs = sell_trading_activity.trade_Qs

        self.rewards_csv(
            ts=kwargs["step_info"]["ts"].isoformat(),
            episode=kwargs["episode"],
            step=kwargs["step_info"]["step"],
            reward=kwargs["reward"].item(),
            buy_rate=kwargs["step_info"]["buy_rate"],
            sell_rate=kwargs["step_info"]["sell_rate"],
            best_buy=kwargs["step_info"]["best_buy"],
            best_sell=kwargs["step_info"]["best_sell"],
            delta_funds=kwargs["step_info"]["delta_funds"],
            delta_inventory=kwargs["step_info"]["delta_inventory"],
            funds=kwargs["step_info"]["funds"],
            inventory=kwargs["step_info"]["inventory"],
            balance=kwargs["step_info"]["balance"],
            buy_match=buy_match,
            buy_Rs=buy_Rs,
            buy_Qs=buy_Qs,
            sell_match=sell_match,
            sell_Rs=sell_Rs,
            sell_Qs=sell_Qs,
        )

    def __call__(self) -> float | None | bool:

        config = self.config  # type: ignore

        def csv_write(locals: dict):
            if self.csv_write:

                try:
                    predicted = {"predicted": self.predicted(state, action)}
                    locals |= predicted
                except RuntimeError:
                    pass

                self.csv_write(
                    **{key: val for key, val in locals.items() if key != "self"}
                )

        mask = (
            lambda: float_tensor([float(done)])
            if hasattr(self.env, "_max_episode_steps")
            and next(episode_step_counter) < self.env._max_episode_steps
            else float_tensor([0.0])
        )

        with MeanReward(
            config
        ) as mean_reward, MeanExpectedProfit() as mean_expected_profit:

            step_counter = it.count(-1)
            step = next(step_counter)
            child_MR = math.inf

            is_logging = constants.CONSOLE != "progress_bar"

            with PlotReward(
                training=config.agent.training, instance_id=config.instance_id
            ) as plot_reward, trange(
                config.agent.episodes,
                disable=is_logging,
                colour="blue" if hasattr(config, "child_process") else "green",
                leave=self.leave_progress_bar,
            ) as t:

                for episode in t:

                    if config.agent.training:
                        increment_training_episodes()

                    self.noise.reset()

                    episode_step_counter = it.count(0)

                    episode_reward = 0

                    state = self.env.reset()

                    state = float_tensor(state).squeeze()

                    with PlotTradingPattern(
                        training=config.agent.training,
                        instance_id=config.instance_id,
                        episode=episode,
                    ) as plot_trading_pattern:

                        start_time = perf_counter()

                        done = False
                        while not done:

                            step = next(step_counter)

                            if step and step % 100 == 0:
                                logging.debug(
                                    f"steps / sec: {step / (perf_counter() - start_time):4.1f}"
                                )

                            if self.training and step < self.exploration:
                                action = float_tensor(self.env.action_space.sample())
                            else:
                                action = self.select_action(state)

                            (next_state, reward, done, step_info,) = self.env.step(  # type: ignore
                                action.cpu().detach().numpy()
                            )

                            plot_trading_pattern(**step_info)

                            reward = float_tensor([reward])

                            next_state = float_tensor(next_state).squeeze()

                            episode_reward += reward

                            if done:
                                plot_reward(reward=reward.item())

                            if self.training:

                                self.replay_buffer + Experience(
                                    state=state,
                                    action=action,
                                    next_state=next_state,
                                    reward=reward,
                                    done=mask(),
                                )

                                if (
                                    step % config.agent.training_interval == 0
                                    and config.agent.batch_size
                                    <= len(self.replay_buffer)
                                ):
                                    self.train(step)

                            else:  # testing only

                                # The following is residual investigation code;
                                # possibly to be removed later
                                if (
                                    hasattr(self, "replay_buffer")
                                    and hasattr(config.agent, "batch_size")
                                    and hasattr(config.agent, "buffer_size")
                                ):

                                    self.replay_buffer + Experience(
                                        state=state,
                                        action=action,
                                        next_state=next_state,
                                        reward=reward,
                                        done=mask(),
                                    )

                                    if config.agent.batch_size <= len(
                                        self.replay_buffer
                                    ):
                                        self.test(step)

                            mean_reward(reward.item())

                            try:
                                mean_expected_profit(step_info["expected_profit"])
                            except KeyError:
                                mean_expected_profit(0)

                            csv_write(locals())

                            state = next_state

                        else:

                            if (
                                self.pytest
                                and self.exploration <= step
                                and self.pytest_successful(episode)
                            ):
                                return True

                            if is_logging:
                                self.log_episode(episode, mean_reward())

                            if self.child_process_required(episode):
                                child_MR = self.launch_child_process()

                        if not is_logging and step % 10000 == 0:
                            t.set_description(
                                f"{config.instance_id:>3} MR={mean_reward():10.4e} "
                                f"MEP= {mean_expected_profit():>10.4e}"
                            )

                else:
                    if self.training and not self.pytest:
                        self.save()

                        self.env.close()

            return child_MR if config.instance_id == 0 else mean_reward()

    def optimize_actor(self, loss: Tensor):

        with self.requires_grad_false(self.critic):

            self.actor_optimizer.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm_(
            #     self.actor.parameters(),
            #     self.config.agent.gradient_clipping.max_norm,
            #     self.config.agent.gradient_clipping.norm_type,
            # )
            self.actor_optimizer.step()

    def optimize_critic(self, loss: Tensor):

        self.critic_optimizer.zero_grad()
        loss.backward()
        # nn.utils.clip_grad_norm_(
        #     self.critic.parameters(),
        #     self.config.agent.gradient_clipping.max_norm,
        #     self.config.agent.gradient_clipping.norm_type,
        # )
        self.critic_optimizer.step()

    @contextmanager
    def requires_grad_false(self, model: nn.Module):
        for p in model.parameters():
            p.requires_grad = False
        yield
        for p in model.parameters():
            p.requires_grad = True
