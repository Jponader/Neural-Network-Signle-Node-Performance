from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from keras. datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras import backend as k

# Helper libraries
import numpy as np
import h5py
import matplotlib.pyplot as plt

import time

test_length = 10000

# Saving Checkpoints while training
import os
checkpoint_path = "training_1/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1)



# print(tf.__version__)

(X_train, y_train), (X_test, y_test) = mnist.load_data()
img_rows , img_cols = 28, 28

if k.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape) 

# Build Model

# Load Model
model= keras.models.load_model('model.h5')


num_category = 10
y_train = keras.utils.to_categorical(y_train, num_category)
y_test = keras.utils.to_categorical(y_test, num_category)

# Source https://github.com/sambit9238/Deep-Learning

"""
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_category, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size = 128)

test_loss, test_acc = model.evaluate(X_test, y_test)

print('Test accuracy:', test_acc)

# Save Weights
model.save_weights('weights.h5')
# model.load_weights()

# Save Entire Model
model.save('model.h5')
# modelNew = keras.models.load_model('model.h5')
"""
imgs = X_test
labels = y_test

def top_k_accuracy(y_true, y_pred, k=1):
    argsorted_y = np.argsort(y_pred)[:,-k:]
    return np.any(argsorted_y.T == y_true.argmax(axis=1), axis=0).mean()


def evaluate():
	pred = model.predict(imgs[:test_length], verbose = 1)
	acc1L = top_k_accuracy(labels[:test_length], pred,1)
	acc5L = top_k_accuracy(labels[:test_length], pred,5)
	print(acc1L)
	print(acc5L)
	return acc1L,acc5L

global_acc1, global_acc5 = evaluate()

diff1 = 0
diff5 = 0
avg1 = 0.00
avg5 = 0.00
test_avg = 0.00
count = 0

def tester():
	global avg1
	global avg5
	global count
	global diff1
	global diff5
	global test_avg

	timer = time.time()
	acc1, acc5 = evaluate()
	end = time.time()

	if acc1 != global_acc1:
		diff1 = diff1 + 1

	if acc5 != global_acc5:
		diff5 = diff5 + 1

	avg1 = avg1 + acc1
	avg5 = avg5 + acc5
	count = count + 1
	test_avg = test_avg + (end - timer)

	return acc1, acc5

# managing weights
model.summary()

stats_Global = h5py.File("stats.h5", "w")

q = 0
for layer in model.layers:

	weights = layer.get_weights()
	weights = np.array(weights)
	stats=stats_Global.create_group(layer.name)


	for x in range(len(weights)):

		if count == 100:
			break


		print("----------------BREAK----------------")
		shape = weights[x].shape
		length = len(shape)
		dataStore = np.empty(shape)
		data5 = np.empty(shape)

		if length == 0:
			continue

		if length == 1:
			for i in range(shape[0]):
				st = weights[x][i]
				weights[x][i] = 0
				layer.set_weights(weights)
				dataStore[i], data5[i] = tester()
				weights[x][i] = st

		if length == 2:
			for i in range(shape[0]):
				for y in range(shape[1]):
					st = weights[x][i][y]
					weights[x][i][y] = 0
					layer.set_weights(weights)
					dataStore[i][y], data5[i][y] = tester()
					weights[x][i][y] = st


		if length == 3:
			for i in range(shape[0]):
				for y in range(shape[1]):
					for j in range(shape[2]):
						st = weights[x][i][y][j]
						weights[x][i][y][j] = 0
						layer.set_weights(weights)
						dataStore[i][y][j], data5[i][y][j] = tester()
						weights[x][i][y][j] = st

		if length == 4:
			for i in range(shape[0]):
				for y in range(shape[1]):
					for j in range(shape[2]):
						for z in range(shape[3]):
							st = weights[x][i][y][j][z]
							weights[x][i][y][j][z] = 0
							layer.set_weights(weights)
							dataStore[i][y][j][x], data5[i][y][j][x] = tester()
							weights[x][i][y][j][z] = st

		if length > 4:
			print("Dimension > 4")
			sys.exit()

		stats.create_dataset(str(q), data=dataStore)
		stats.create_dataset(str(q+.5), data=data5)
		q = q + 1

stats_Global.close()

print("Global accuracy")
print(global_acc1)
print(global_acc5)

print(count)
print("AVG Accuracy1")
print(avg1/count)
print(diff1)


print("AVG Accuracy5")
print(avg5/count)
print(diff5)

