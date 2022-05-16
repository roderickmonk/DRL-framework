import gym
import numpy as np
import torch
import torch.nn as nn
from constants import DEVICE
from torch import tensor
from torch.distributions import MultivariateNormal
from torch.optim import Adam
import itertools as it
import os


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std):
        super().__init__()
        # action mean range -1 to 1
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, action_dim),
            nn.Tanh(),
        )
        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )
        self.action_var = torch.full((action_dim,), action_std * action_std).to(DEVICE)

        self.to(DEVICE)

    def forward(self):
        raise NotImplementedError

    def act(self, state, memory):

        action_mean = self.actor(state)
        cov_mat = torch.diag(self.action_var).to(DEVICE)

        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)

        return action.detach()

    def evaluate(self, state, action):
        action_mean = self.actor(state)

        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(DEVICE)

        dist = MultivariateNormal(action_mean, cov_mat)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.critic(state)

        # print(f"{state_value.dtype=}")
        # print(f"{torch.squeeze(state_value).dtype=}")
        # print(f"{action_logprobs.dtype=}")
        # print(f"{dist_entropy.dtype=}")

        return action_logprobs, torch.squeeze(state_value), dist_entropy


class PPO:
    def __init__(
        self, *, state_dim, action_dim, action_std, lr, gamma, K_epochs, eps_clip
    ):
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = ActorCritic(state_dim, action_dim, action_std).to(DEVICE)
        self.optimizer = Adam(self.policy.parameters(), lr=lr)

        self.policy_old = ActorCritic(state_dim, action_dim, action_std).to(DEVICE)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def select_action(self, state, memory):

        # print(f"1. {state=}")
        # print(f"{state.shape=}")
        # print(f"{type(state)=}")

        state = (
            state.reshape(1, -1)
            if isinstance(state, torch.Tensor)
            else tensor(state, device=DEVICE).reshape(1, -1)
        )

        state = state.to(DEVICE)
        # print(f"2. {state=}")
        # print(f"{state.shape=}")
        # print(f"{type(state)=}")

        return self.policy_old.act(state, memory).cpu().data.numpy().flatten()

    def update(self, memory):
        # Monte Carlo estimate of rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(
            reversed(memory.rewards), reversed(memory.is_terminals)
        ):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = tensor(rewards, device=DEVICE)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        old_states = torch.squeeze(torch.stack(memory.states).to(DEVICE), 1).detach()
        old_actions = torch.squeeze(torch.stack(memory.actions).to(DEVICE), 1).detach()
        old_logprobs = (
            torch.squeeze(torch.stack(memory.logprobs), 1).to(DEVICE).detach()
        )

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(
                old_states, old_actions
            )

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # print(f"{ratios.dtype=}")
            # print(f"{rewards.dtype=}")
            # print(f"{state_values.dtype=}")

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            # print(f"{advantages.dtype=}")

            surr1 = ratios * advantages
            surr2 = (
                torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            )

            # print(f"{surr1.dtype=}")
            # print(f"{surr2.dtype=}")

            loss = (
                -torch.min(surr1, surr2)
                + 0.5 * self.MseLoss(state_values, rewards)
                - 0.01 * dist_entropy
            ).to(torch.float32)

            # print(f"{loss.dtype=}")

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().to(torch.float32).backward()
            self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())


def main():
    ############## Hyperparameters ##############
    env_name = "Pendulum-v1"
    solved_reward = -0  # stop training if avg_reward > solved_reward
    log_interval = 20  # print avg reward in the interval
    max_episodes = 10000  # max training episodes

    update_timestep = 4000  # update policy every n timesteps
    action_std = 0.5  # constant std for action distribution (Multivariate Normal)
    K_epochs = 80  # update policy for K epochs
    eps_clip = 0.2  # clip parameter for PPO
    gamma = 0.99  # discount factor

    lr = 0.0003

    # creating environment
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]  # type: ignore
    action_dim = env.action_space.shape[0]  # type: ignore

    memory = Memory()
    ppo = PPO(
        state_dim=state_dim,
        action_dim=action_dim,
        action_std=action_std,
        lr=lr,
        gamma=gamma,
        K_epochs=K_epochs,
        eps_clip=eps_clip,
    )

    time_step = 0

    # training loop
    for episode in range(1, max_episodes + 1):

        episode_length = it.count(0)
        episode_reward = 0

        state, done = env.reset(), False

        while not done:
            time_step += 1
            # Running policy_old:
            action = ppo.select_action(state, memory)
            state, reward, done, _ = env.step(action)

            # print(f"{state=}")
            # print(f"{type(state)=}")
            # print(f"{state.dtype=}")
            state = tensor(state)
            reward = tensor(reward)

            # print(f"{state.dtype=}")
            # print(f"{reward.dtype=}")
            # print(f"{done.dtype=}")

            # Saving reward and is_terminals:
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            # update if its time
            if time_step % update_timestep == 0:
                ppo.update(memory)
                memory.clear_memory()
                time_step = 0

            episode_reward += reward

            next(episode_length)

        # stop training if avg_reward > solved_reward
        if episode_reward > solved_reward:
            print(f"########## Solved! ########## {episode_reward=}")
            break

        # logging
        if episode % log_interval == 0:

            print(f"Episode: {episode}\t " f"Episode Reward: {episode_reward:>10.3f}")


if __name__ == "__main__":
    main()

