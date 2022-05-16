import pandas as pd
from dataset import Dataset
import numpy as np

full_set = Dataset().read(source="/shared/co3/datasets/small_sell_ob_training_log.json")
full_set = full_set[0: (len(full_set) - len(full_set) % 50)]   # the value of the modulus is based on the inteded batch size
actions = np.random.uniform(0.02, 0.98, size = len(full_set))
full_set['action']=actions

#targets need to be calculated based on reward function and the newly added actions
states = full_set['sell_ob_vector']
targets = np.empty(shape = len(full_set))
for i in range(len(full_set)):
    state = 10**np.array(states.iloc[i])-1e-12
    action = actions[i]
    targets[i] = (-((2 * action - 1) ** 2) + 1 ** 2) * np.sum(state) #parabola between 0 and 1 *sum(state)

full_set['target'] = targets

Dataset().persist(source=full_set, dest="/shared/co3/datasets/eth_with_actions_targets_training", overwrite=True)
