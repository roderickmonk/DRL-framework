import datetime
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import json
import pandas as pd

upper_action = 0.98
lower_action = 0.02
upper_reward_limit = 1
lower_reward_limit = 0
state_size = 6
allseed = 7
np.random.seed(allseed)
torch.manual_seed(allseed)
input_length = state_size + 1


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_length, 250)
        self.fc2 = nn.Linear(250, 200)
        self.fc3 = nn.Linear(200, 150)
        self.fc4 = nn.Linear(150, 1)

    def forward(self, state, action):
        if len(action.unsqueeze(-1).detach().numpy()) == 1:
            x = torch.from_numpy(np.append(state, action.unsqueeze(-1), axis=0))
        elif len(action.unsqueeze(-1).detach().numpy()) > 1:
            x = torch.from_numpy(np.append(state, action.unsqueeze(-1), axis=1))
        else:
            raise ValueError()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


net = Net()

# load network and weights/biases
checkpoint = torch.load(
    "/shared/co3/networks/ddpg_parabola_tool_replicav5_untrained.pt"
)
net.load_state_dict(checkpoint["critic"])


def de_normalize(data):
    return 10 ** (data) - 1e-12


# loss function and optimizer

criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.00001)

# download datasets
with open("/shared/co3/datasets/evaluate/eth_with_actions_targets_test.json") as data:
    test_dataset = json.load(data)

with open(
    "/shared/co3/datasets/evaluate/eth_with_actions_targets_training.json"
) as data:
    train_dataset = json.load(data)

# transform data from dataset into inputs
train_vector = torch.from_numpy(np.array(train_dataset["sell_ob_vector"]))
train_states_denorm = np.array(
    [de_normalize(train_vector[i].detach().numpy()) for i in range(len(train_vector))]
)
train_actions = torch.from_numpy(np.array(train_dataset["action"]))
train_targets = torch.from_numpy(np.array(train_dataset["target"]))

# other variables required for testing
test_vector = torch.from_numpy(np.array(test_dataset["sell_ob_vector"]))
test_actions = torch.from_numpy(np.array(test_dataset["action"]))
test_targets = torch.from_numpy(np.array(test_dataset["target"]))

# test function
def test_df(test_states, test_actions, test_labels, net, df):
    error = 0
    results_df = df
    with torch.no_grad():
        for i in [0]:
            out = net(test_states.float(), test_actions.float())
            error = torch.mean(torch.abs(torch.transpose(out.data, 0, 1) - test_labels))
            for i in range(len(test_actions)):
                results_df = results_df.append(
                    {
                        "Action": test_actions[i].detach().numpy(),
                        "Output": out[i].detach().numpy()[0],
                        "Target": test_labels[i].detach().numpy(),
                        "State_Sum": np.sum(de_normalize(np.array(test_vector[i]))),
                    },
                    ignore_index=True,
                )
    return results_df


# training with dataframe creation and testing loop

results_df = pd.DataFrame(columns=["Action", "Output", "Target", "State_Sum"])
df_sum = [np.sum(train_states_denorm[0:1000][i]) for i in range(1000)]

for epoch in range(1):  # loop over the dataset only once for this
    running_loss = 0.0
    for i in range(1000):
        # zero the parameter gradients
        optimizer.zero_grad()
        outputs = net(train_vector[i].float(), train_actions[i].float()).squeeze(-1)
        loss = criterion(outputs, np.log10(train_targets[i]).float())  # type:ignore
        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.item()
        if (i + 1) % 1 == 0:  # print every 5000 mini-batches
            print("[%d, %5d] loss: %.6f" % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
        if (i + 1) % 1 == 0:  # print every 1000 mini-batches
            print(
                "[%d, %5d] MAE: %.6f"
                % (
                    epoch + 1,
                    i + 1,
                    np.mean(
                        np.abs(
                            outputs.detach().numpy()
                            - np.log10(train_targets[i].detach().numpy())
                        )
                    ),
                )
            )
        data = pd.DataFrame(
            {
                "Action": train_actions[i].detach().numpy(),
                "Output": outputs.detach().numpy(),
                "Target": train_targets[i].detach().numpy(),
                "State_Sum": np.array(df_sum[i]),
            },
            index=range(1),
        )
        results_df = results_df.append(data)
        testing_df = pd.DataFrame(columns=["Action", "Output", "Target", "State_Sum"])
        test_results = test_df(
            test_states=torch.from_numpy(np.array(test_vector)),
            test_actions=torch.from_numpy(np.array(test_actions)),
            test_labels=np.log10(test_targets),
            net=net,
            df=testing_df,
        )
        dt = datetime.datetime.now()
        ii = i + 1
        test_results.to_csv(
            f"rewards/ddpg/manual_tests/v5_test/test_frame{dt}-{ii}.csv"
        )
        print("Testing Complete")

print("Finished Training")

results_df.to_csv(
    "rewards/ddpg/manual_tests/v5_test/training_frame{}.csv".format(
        datetime.datetime.now()
    )
)


# save network
torch.save(
    {"model_state_dict": net.state_dict(),}, "networks/manual_tool/parabolav5.pt"
)

