import numpy as np
from sklearn import preprocessing

# import os
# print(os.getcwd())

# 1. Extract the data from the csv
#-----------------------------------------

raw_csv_data = np.loadtxt('Audiobooks_data.csv', delimiter = ',')

unscaled_inputs_all = raw_csv_data[:,1:-1]
targets_all = raw_csv_data[:,-1]
print("Total No of Samples", targets_all.shape[0])

# 2. Balance the dataset
#-----------------------------------------
print("..Balance the dataset....")
num_one_targets = int(np.sum(targets_all))
zero_targets_counter = 0
indices_to_remove = []

for i in range(targets_all.shape[0]):
    if targets_all[i] == 0:
        zero_targets_counter += 1
        if zero_targets_counter > num_one_targets:
            indices_to_remove.append(i)

unscaled_inputs_equal_priors = np.delete(unscaled_inputs_all, indices_to_remove, axis=0)
targets_equal_priors = np.delete(targets_all, indices_to_remove, axis=0)

print("Data Items After balancing: ", targets_equal_priors.shape[0])


# 3. Standardize the inputs
#-----------------------------------------
scaled_inputs = preprocessing.scale(unscaled_inputs_equal_priors)

scaled_inputs = preprocessing.scale(unscaled_inputs_equal_priors)

# 4. Shuffle the data
#-----------------------------------------
shuffled_indices = np.arange(scaled_inputs.shape[0])
np.random.shuffle(shuffled_indices)
shuffled_inputs = scaled_inputs[shuffled_indices]
shuffled_targets = targets_equal_priors[shuffled_indices]


# 5. Split the dataset into train, validation, and test
# -----------------------------------------
samples_count = shuffled_inputs.shape[0]
# print(samples_count)
#
# print("Data Type", type(samples_count))


train_samples_count = int(0.8*samples_count)  # type: int
validation_samples_count = int(0.1*samples_count)
test_samples_count = samples_count - train_samples_count - validation_samples_count

train_inputs = shuffled_inputs[:train_samples_count]
train_targets = shuffled_targets[:train_samples_count]

validation_inputs = shuffled_inputs[train_samples_count:train_samples_count+validation_samples_count]
validation_targets = shuffled_targets[train_samples_count:train_samples_count+validation_samples_count]

test_inputs = shuffled_inputs[train_samples_count+validation_samples_count:]
test_targets = shuffled_targets[train_samples_count+validation_samples_count:]## Ratio of positive and negative sample data for each train, validation and test data sets

print("Positive------Total--------------Ratio")
print("P: ", np.sum(train_targets), "Tot: ", train_samples_count, "Ratio: ", np.sum(train_targets) / train_samples_count)
print("P: ", np.sum(validation_targets), "Tot: ",validation_samples_count, "Ratio: ", np.sum(validation_targets) / validation_samples_count)
print("P: ", np.sum(test_targets), "Tot: ", test_samples_count, "Ratio: ", np.sum(test_targets) / test_samples_count)

# 6. Save the three datasets in *.npz
#-----------------------------------------
np.savez('Audiobooks_data_train', inputs=train_inputs, targets=train_targets)
np.savez('Audiobooks_data_validation', inputs=validation_inputs, targets=validation_targets)
np.savez('Audiobooks_data_test', inputs=test_inputs, targets=test_targets)