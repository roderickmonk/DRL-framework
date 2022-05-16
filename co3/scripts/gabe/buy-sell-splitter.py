import pandas as pd
from dataset import Dataset

# Dataset().ls()
full_set = Dataset().read(source="FullBalanceSNT")

# randomize then split the data sets
random_set = full_set.sample(frac=1)

# reindex
random_set.index = range(0, len(random_set))  # type:ignore

# get buy and sell side ITs
buy_it = random_set["it"].str[0]
buy_it = buy_it[buy_it > 0]
sell_it = random_set["it"].str[1]
sell_it = sell_it[sell_it > 0]

# get buy and sell side states
buy_state = random_set.state[buy_it > 0]
sell_state = random_set.state[sell_it > 0]

# recombine state and ITs
buy_set = pd.concat([buy_state, buy_it], axis=1)
sell_set = pd.concat([sell_state, sell_it], axis=1)

# save to file
Dataset().persist(source=buy_set, dest="buy_snt", overwrite=True)
Dataset().persist(source=sell_set, dest="sell_snt", overwrite=True)

buy_set_test = buy_set[0:10000]
buy_set_training = buy_set[10000 : len(buy_set) + 1]
sell_set_test = sell_set[0:10000]
sell_set_training = sell_set[10000 : len(buy_set) + 1]

Dataset().persist(
    source=buy_set_test,  # type:ignore
    dest="buy_snt_test",
    overwrite=True,
)
Dataset().persist(
    source=sell_set_test,  # type:ignore
    dest="sell_snt_test",
    overwrite=True,
)
Dataset().persist(
    source=buy_set_training,  # type:ignore
    dest="buy_snt_training",
    overwrite=True,
)
Dataset().persist(
    source=sell_set_training,  # type:ignore
    dest="sell_snt_training",
    overwrite=True,
)
