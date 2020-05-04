from keras.datasets import boston_housing
from keras import models
from keras import layers
import numpy as np

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

print(train_data.shape)

print(test_data.shape)

mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
test_data -= mean
test_data /= std

#It would be problematic to feed into a neural network values that all take wildly different ranges. The network might be able to automatically adapt to such heterogeneous data, but it would definitely make learning more difficult. A widespread best practice to deal with such data is to do feature-wise normalization: for each feature in the input data (a column in the input data matrix), you subtract the mean of the feature and divide by the standard deviation, so that the feature is centered around 0 and has a unit standard deviation. This is easily done in Numpy.


def build_model():
	model = models.Sequential()
	model.add(layers.Dense(64, activation='relu',
	input_shape=(train_data.shape[1],)))
	model.add(layers.Dense(64, activation='relu'))
	model.add(layers.Dense(1))
	model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
	return model


k = 4
num_val_samples = len(train_data) // k
num_epochs = 100
all_scores = []


for i in range(k):
	print('processing fold #', i)
	val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
	val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
	partial_train_data = np.concatenate([train_data[:i * num_val_samples],train_data[(i + 1) * num_val_samples:]],axis=0)
	partial_train_targets = np.concatenate([train_targets[:i * num_val_samples],train_targets[(i + 1) * num_val_samples:]],axis=0)
	model = build_model()
	model.fit(partial_train_data, partial_train_targets,
	epochs=num_epochs, batch_size=1, verbose=0)
	val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
	print(val_mse)
	print(val_mae)
	all_scores.append(val_mae)
