import random

import pandas as pd
from dataset import Dataset

random.seed(7)
# Dataset().ls()

full_set_bal = Dataset(evaluate=False).read(source="FullBalanceSNT")
full_set_eval = Dataset(evaluate=True).read(source="FULLSNT20180220")

# randomize then split the data sets
random_set_bal = full_set_bal.sample(frac=1)
random_set_eval = full_set_eval.sample(frac=1)

#get buy and sell side ITs
buy_it = random_set_bal['it'].str[0]
buy_it = buy_it[buy_it>0]
sell_it = random_set_bal['it'].str[1]
sell_it = sell_it[sell_it>0]

buy_index = buy_it.index
sell_index = sell_it.index

#get buy and sell side states
buy_state = random_set_bal.state[buy_index]
sell_state = random_set_bal.state[sell_index]

#recombine state and ITs
buy_set = pd.concat([buy_state, buy_it], axis = 1)
sell_set = pd.concat([sell_state, sell_it], axis = 1)

# save to file
Dataset(evaluate=False).persist(source=buy_set, dest="new_buy_snt", overwrite=True)
Dataset(evaluate=False).persist(source=sell_set, dest="new_sell_snt", overwrite=True)

#create test and training sets for buy and sell sides
buy_set_test = buy_set[0:10000]
buy_set_training = buy_set[10000 : len(buy_set) + 1]
sell_set_test = sell_set[0:10000]
sell_set_training = sell_set[10000 : len(buy_set) + 1]

#get new index to have same test set between evaluate and balance sets
test_index = buy_index[0:10000].union(sell_index[0:10000])

#get test set with index and exclude that index from training set
new_snt_test = random_set_eval.loc[test_index]
bad_index = random_set_eval.index.isin(test_index)
new_snt_training = random_set_eval[~bad_index]

#save to file
Dataset(evaluate=True).persist(source=buy_set_test, dest="new_buy_snt_test", overwrite=True)
Dataset(evaluate=True).persist(source=sell_set_test, dest="new_sell_snt_test", overwrite=True)
Dataset(evaluate=True).persist(source=buy_set_training, dest="new_buy_snt_training", overwrite=True)
Dataset(evaluate=True).persist(source=sell_set_training, dest="new_sell_snt_training", overwrite=True)
Dataset(evaluate=True).persist(source=new_snt_test, dest="new_snt_test", overwrite=True)
Dataset(evaluate=True).persist(source=new_snt_training, dest="new_snt_training", overwrite=True)
