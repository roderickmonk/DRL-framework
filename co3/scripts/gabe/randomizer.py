from dataset import Dataset

full_set = Dataset(evaluate=True).read(source="ALTSNT20180220")

# randomize then split the data sets
random_set = full_set.sample(frac=1)
test_set = random_set[0:21440]
training_set = random_set[21441 : len(random_set) + 1]

# reindex
test_set.index = range(0, len(test_set))
training_set.index = range(0, len(training_set))

# save to file
Dataset(evaluate=True).persist(source=test_set, dest="alt_test_set_snt", overwrite=True)
Dataset(evaluate=True).persist(
    source=training_set, dest="alt_training_set_snt", overwrite=True
)
