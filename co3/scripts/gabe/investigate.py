#!/usr/bin/env python

import json
import logging
import os
import random
import traceback

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import util
from constants import Anomaly, ConfigurationError
from omegaconf import OmegaConf
from torch import tensor

upper_action = 0.98
lower_action = 0.02
upper_reward_limit = 1
lower_reward_limit = 0
state_size = 6
allseed = 7
np.random.seed(allseed)
torch.manual_seed(allseed)
random.seed(allseed)

input_length = state_size + 1

torch.set_printoptions(precision=10, sci_mode=True)


def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.manual_seed(7)
        nn.init.xavier_uniform_(m.weight, gain=1)
        nn.init.constant_(m.bias, 0)


class Net(nn.Module):
    """"""

    def __init__(self):

        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_length, 250)
        self.fc2 = nn.Linear(250, 200)
        self.fc3 = nn.Linear(200, 150)
        self.fc4 = nn.Linear(150, 1)

        self.apply(weights_init_)

        # for name, param in self.named_parameters():
        #     print(f"{name}:\n{param}")

    def forward(self, state, action):

        if action.ndim == 2:
            x = torch.cat([state, action], 1)
        elif action.ndim == 1:
            x = torch.cat([state, action.unsqueeze(-1)], 1)
        else:
            raise Anomaly(f"action wrong shape: {action.shape}")

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x


def main():

    omegaconf = OmegaConf.create(
        {"csv_path": "rewards/investigate"} | dict(OmegaConf.from_cli())
    )

    net = Net()

    def de_normalize(data):
        return 10 ** (data) - 1e-12

    # download datasets
    with open("datasets/evaluate/eth_with_actions_targets_test.json") as data:
        test_dataset = json.load(data)

    with open("datasets/evaluate/eth_with_actions_targets_training.json") as data:
        train_dataset = json.load(data)

    # transform data from dataset into inputs
    train_vector = tensor(train_dataset["sell_ob_vector"], dtype=torch.float32)
    train_actions = tensor(train_dataset["action"], dtype=torch.float32)
    train_targets = tensor(train_dataset["target"], dtype=torch.float32)

    test_vector = tensor(test_dataset["sell_ob_vector"], dtype=torch.float32)
    test_actions = tensor(test_dataset["action"], dtype=torch.float32)
    test_targets = tensor(test_dataset["target"], dtype=torch.float32)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.00001)

    os.makedirs(f"{omegaconf.csv_path}", mode=0o775, exist_ok=True)

    for i in range(5):

        optimizer.zero_grad()

        vector = train_vector[i].view([1, -1])
        action = train_actions[i].view([1])

        output = net(vector, action)

        loss = criterion(output.view([]), torch.log10(train_targets[i]))
        loss.backward()
        optimizer.step()

        test_results = util.test_df(
            states=test_vector,
            actions=test_actions,
            labels=torch.log10(test_targets),
            net=net,
        )
        test_results.to_csv(f"{omegaconf.csv_path}/testing_frame-{i}.csv")


if __name__ == "__main__":

    try:
        main()

    except KeyboardInterrupt:
        logging.fatal("KeyboardInterrupt")

    except ConfigurationError as err:
        traceback.print_stack()
        logging.fatal(err)

    except Anomaly as err:
        traceback.print_stack()
        logging.fatal(err)

    else:
        logging.warning("✨That's✨All✨Folks✨")
