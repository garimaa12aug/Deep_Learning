from keras.datasets import imdb
from keras import models
from keras import layers

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
#The argument num_words=10000 means you’ll only keep the top 10,000 most frequently occurring words in the training data
#The variables train_data and test_data are lists of reviews; each review is a list of word indices, train_labels and test_labels are lists of 0s and 1s, where 0 stands for negative and 1 stands for positive movie review

print(train_data[0])

print(train_labels[0])

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))


model.compile(optimizer='rmsprop', loss='binary_crossentropy',metrics=['accuracy'])

model.fit(train_data, train_labels, epochs=4, batch_size=512)
results = model.evaluate(test_data, test_labels)
print('training result: ',results)


#x_val = train_data[:10000]
#partial_x_train = train_data[10000:]
#y_val = train_labels[:10000]
#partial_y_train = train_labels[10000:]

#history = model.fit(partial_x_train,partial_y_train,epochs=20,batch_size=512,validation_data=(x_val, y_val))
#results = model.evaluate(test_data, test_labels)
#print('validation result: ',results)

