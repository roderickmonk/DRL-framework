import logging
import sys
from typing import Any

import torch
import torch.nn as nn
import util
from agent.agent import AbstractAgent
from constants import DEVICE
from networks.fully_connected import FullyConnected
from mean_expected_profit import MeanExpectedProfit
from mean_reward import MeanReward
from rewards_csv import ContinuousRewardsCsv
from torch import Tensor, tensor
from torch.distributions import MultivariateNormal
from pydantic_config import PpoProcessConfig
from project_functions import initialize_weights

if torch.cuda.is_available():
    torch.cuda.empty_cache()


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std):
        super(ActorCritic, self).__init__()

        self.action_dim = action_dim
        self.action_var = torch.full((action_dim,), action_std * action_std).to(DEVICE)

        activation = nn.ReLU()

        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256),
            activation,
            nn.Linear(256, 128),
            activation,
            nn.Linear(128, 64),
            activation,
            nn.Linear(64, 32),
            activation,
            nn.Linear(32, action_dim),
            activation,
        )

        # self.actor = FullyConnected(
        #     input_size=state_dim,
        #     hidden_dims=[256, 128, 64, 32],
        #     activations=[activation],
        # )

        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256),
            activation,
            nn.Linear(256, 128),
            activation,
            nn.Linear(128, 64),
            activation,
            nn.Linear(64, 32),
            activation,
            nn.Linear(32, 1),
        )

        # self.critic = FullyConnected(
        #     input_size=state_dim,
        #     hidden_dims=[256, 128, 64, 32],
        #     output_size=1,
        #     activations=[activation],
        # )

        initialize_weights(self)

        self.to(DEVICE)

    def set_action_std(self, new_action_std):

        self.action_var = torch.full(
            (self.action_dim,), new_action_std * new_action_std
        ).to(DEVICE)

    def forward(self):
        raise NotImplementedError

    def act(self, state):

        action_mean = self.actor(state)
        cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
        dist = MultivariateNormal(action_mean, cov_mat)

        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action):

        action_mean = self.actor(state)

        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(DEVICE)
        dist = MultivariateNormal(action_mean, cov_mat)

        # For Single Action Environments.
        if self.action_dim == 1:
            action = action.reshape(-1, self.action_dim)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy


class Agent(AbstractAgent):
    def __init__(self, *, config: PpoProcessConfig, env):

        super().__init__(config=config, env=env)

        self.config = config

        self.state_size = self.env.observation_space.shape[0]  # type:ignore
        self.action_size = self.env.action_space.shape[0]  # type:ignore

        self.action_space_low = self.env.action_space.low[0]  # type:ignore
        self.action_space_high = self.env.action_space.high[0]  # type:ignore

        self.pytest = "pytest" in sys.modules
        self.pytest_success_threshold = -250

        self.rewards_csv = ContinuousRewardsCsv(config=self.config)

        self.action_std = config.agent.action_std_init

        self.gamma = config.agent.gamma
        self.eps_clip = config.agent.eps_clip
        self.K_epochs = config.agent.K_epochs

        self.buffer = RolloutBuffer()

        state_dim = self.env.observation_space.shape[0]  # type:ignore
        action_dim = self.env.action_space.shape[0]  # type:ignore

        self.policy = ActorCritic(state_dim, action_dim, self.action_std).to(DEVICE)
        self.actor = self.policy.actor
        self.critic = self.policy.critic

        self.optimizer = torch.optim.Adam(
            [
                {"params": self.policy.actor.parameters(), "lr": config.agent.actor_lr},
                {
                    "params": self.policy.critic.parameters(),
                    "lr": config.agent.critic_lr,
                },
            ]
        )

        self.policy_old = ActorCritic(state_dim, action_dim, self.action_std).to(DEVICE)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):

        self.action_std = new_action_std
        self.policy.set_action_std(new_action_std)
        self.policy_old.set_action_std(new_action_std)

    def decay_action_std(self, action_std_decay_rate, min_action_std):

        self.action_std = self.action_std - action_std_decay_rate
        self.action_std = round(self.action_std, 4)
        if self.action_std <= min_action_std:
            self.action_std = min_action_std
            logging.error(
                f"setting actor output action_std to min_action_std : {self.action_std}"
            )
        else:
            logging.error(f"setting actor output action_std to: {self.action_std}")
        self.set_action_std(self.action_std)

    def select_action(self, state):

        with torch.no_grad():
            action, action_logprob = self.policy_old.act(state)

        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)

        action = torch.clamp(action, self.action_space_low, self.action_space_high)

        return action.detach()

    def train(self, *args):

        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(
            reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)
        ):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalize the rewards
        rewards = tensor(rewards, dtype=torch.float32, device=DEVICE)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        old_states = (
            torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(DEVICE)
        )
        old_actions = (
            torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(DEVICE)
        )
        old_logprobs = (
            torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(DEVICE)
        )

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(
                old_states, old_actions
            )

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = (
                torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            )

            # final loss of clipped objective PPO
            loss = (
                -torch.min(surr1, surr2)
                + 0.5 * self.MseLoss(state_values, rewards)
                - 0.01 * dist_entropy
            )

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

    def predicted(self, state: Tensor, action: Tensor):
        return 0.0

    def __call__(self):

        config: PpoProcessConfig = self.config  # type:ignore

        env: Any = self.env

        # is_logging = constants.CONSOLE != "progress_bar"
        is_logging = True

        def csv_write():

            self.rewards_csv(
                episode=episode,
                step=step,
                reward=reward.item(),
                action=action.item(),
                state=state,
                expected_profit=step_info["expected_profit"] if not self.pytest else 0,
                predicted=self.predicted(state, action),
            )

        step = 0

        with MeanReward(
            config
        ) as mean_reward, MeanExpectedProfit() as mean_expected_profit:

            for episode in range(config.agent.episodes):

                state = tensor(env.reset(), dtype=torch.float32, device=DEVICE)
                episode_reward = 0

                done = False
                while not done:

                    # select action with policy
                    action = self.select_action(state)
                    state, reward, done, step_info = env.step(
                        action.cpu().numpy().flatten()
                    )

                    state = tensor(state, dtype=torch.float32, device=DEVICE)

                    # saving reward and is_terminals
                    self.buffer.rewards.append(reward)
                    self.buffer.is_terminals.append(done)

                    step += 1
                    episode_reward += reward

                # if continuous action space; then decay action std of ouput action distribution
                if step % config.agent.action_std_decay_freq == 0:
                    self.decay_action_std(
                        config.agent.action_std_decay_rate, config.agent.min_action_std
                    )

                if episode != 0 and episode % 5 == 0:
                    self.train()

                if episode != 0 and episode % config.misc.log_interval == 0:
                    logging.debug(f"{episode=}, {episode_reward=}")

                if config.misc.generate_csv:
                    csv_write()

                if self.pytest and self.pytest_successful(episode):
                    return True

                if is_logging:
                    self.log_episode(episode, mean_reward())

                if self.child_process_required(episode):
                    child_MR = self.launch_child_process()

        env.close()

    def log_episode(self, episode, MR):

        if (
            episode != 0
            and (
                episode % self.config.misc.log_interval == 0
                or episode == self.config.agent.episodes - 1
            )
        ) and not self.pytest:
            logging.info(
                f"Episode: {episode:5}, "
                f"Mean Reward: {MR:10.6f} "
                f"{util.log_interval()}"
            )

    def pytest_successful(self, episode, eval_interval=10, eval_episodes=10):

        env: Any = self.env

        if all([self.pytest, episode, episode % eval_interval]) == 0:

            with torch.no_grad():

                eval_rewards = 0

                for _ in range(eval_episodes):

                    state = tensor(env.reset(), dtype=torch.float32, device=DEVICE)

                    done = False
                    while not done:

                        with torch.no_grad():
                            action, _ = self.policy_old.act(state)
                            state, reward, done, _ = env.step(
                                action.cpu().numpy().flatten()
                            )
                            state = tensor(state, dtype=torch.float32, device=DEVICE)
                            eval_rewards += reward

                else:

                    eval_rewards /= eval_episodes

                    logging.error(f"{episode=:>5}, {eval_rewards=:>8.1f}")

                    if self.test_evaluation(eval_rewards):
                        logging.critical(f"Environment Solved")
                        return True

    def test_evaluation(self, evaluation_rewards):

        try:
            if (
                evaluation_rewards > self.env.spec.reward_threshold * 1.1  # type:ignore
            ):  # x 1.1 because of small eval_episodes
                return True

        except TypeError:
            if evaluation_rewards > self.pytest_success_threshold:
                print(f"{evaluation_rewards=}, {self.pytest_success_threshold=}")
                return True

        return False
