from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

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

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']



# Shows how and Image is broken down
# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()

train_images = train_images / 255.0

test_images = test_images / 255.0

# Shows some sample Images and Lables
# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()

# Build Model

# Load Model
model= keras.models.load_model('model.h5')
"""

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5, callbacks = [cp_callback])

test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)

# Save Weights
# model.save_weights('weights.h5')
# model.load_weights()

# Save Entire Model
model.save('model.h5')
"""
# modelNew = keras.models.load_model('model.h5')

imgs = test_images
labels = test_labels


#-------------------------------------------------------------------------------#

def top_k_accuracy(y_true, y_pred, k=1):
    argsorted_y = np.argsort(y_pred)[:,-k:]
    return np.any(argsorted_y.T == y_true, axis=0).mean()


def accuracy_diff(y_true, y_pred):
	diff = 0
	count = 0
	for item in y_true:
		diff = y_pred[count][item] - global_pred[count][item] + diff
		count = count + 1

	return diff/count	

def tester():
	global avg1
	global avg5
	global count
	global diff1
	global diff5
	global avgDiff

	pred = model.predict(imgs[:test_length], verbose = 1)
	acc1 = top_k_accuracy(labels[:test_length], pred,1)
	acc5 = top_k_accuracy(labels[:test_length], pred,5)
	accdiff = accuracy_diff(labels[:test_length], pred)
	print(accdiff)
	print(acc1)
	print(acc5)
	AddData( acc1, acc5, accdiff)

	if acc1 != global_acc1:
		diff1 = diff1 + 1

	if acc5 != global_acc5:
		diff5 = diff5 + 1

	avg1 = avg1 + acc1
	avg5 = avg5 + acc5
	avgDiff = avgDiff + accdiff
	count = count + 1
	return acc1, acc5

def AddData(acc1L,acc5L, avgDiff):
	global dataStore
	global data5
	global accdiff
	global rowcount

	dataStore[rowcount] = acc1L
	data5[rowcount] = acc5L
	accdiff[rowcount] = avgDiff
	rowcount = rowcount + 1



def create(size):
	global dataStore
	global data5
	global accdiff
	global rowcount

	dataStore = np.empty(size)
	data5 = np.empty(size)
	accdiff = np.empty(size)
	rowcount = 0

def store(shape):
	global dataStore
	global data5
	global accdiff
	global q


	dataStore = dataStore.reshape(shape)
	data5 = data5.reshape(shape)
	accdiff = accdiff.reshape(shape)
	stats.create_dataset(str(q), data=dataStore)
	stats.create_dataset(str(q+.5), data=data5)
	stats.create_dataset("accDiff" + str(q), data=accdiff)
	q = q + 1






#-------------------------------------------------------------------------------#


global_pred = model.predict(imgs[:test_length], verbose = 1)
global_acc1 = top_k_accuracy(labels[:test_length], global_pred,1)
global_acc5 = top_k_accuracy(labels[:test_length], global_pred,5)

diff1 = 0
diff5 = 0
avg1 = 0.00
avg5 = 0.00
count = 0
avgDiff = 0.00

# managing weights
model.summary()

stats_Global = h5py.File("stats.h5", "w")

dataStore = np.empty(0)
data5 = np.empty(0)
accdiff = np.empty(0)
rowcount = 0;

q = 0
for layer in model.layers:

	weights = layer.get_weights()
	weights = np.array(weights)
	stats=stats_Global.create_group(layer.name)


	for x in range(len(weights)):

		print("----------------BREAK----------------")
		shape = weights[x].shape
		size = weights[x].size
		length = len(shape)
		create(size)

		if length == 0:
			continue

		if length == 1:
			for i in range(shape[0]):
				st = weights[x][i]
				weights[x][i] = 0
				layer.set_weights(weights)
				tester()
				weights[x][i] = st

		if length == 2:
			for i in range(shape[0]):
				for y in range(shape[1]):
					st = weights[x][i][y]
					weights[x][i][y] = 0
					layer.set_weights(weights)
					tester()
					weights[x][i][y] = st


		if length == 3:
			for i in range(shape[0]):
				for y in range(shape[1]):
					for j in range(shape[2]):
						st = weights[x][i][y][j]
						weights[x][i][y][j] = 0
						layer.set_weights(weights)
						tester()
						weights[x][i][y][j] = st

		if length == 4:
			for i in range(shape[0]):
				for y in range(shape[1]):
					for j in range(shape[2]):
						for z in range(shape[3]):
							st = weights[x][i][y][j][z]
							weights[x][i][y][j][z] = 0
							layer.set_weights(weights)
							tester()
							weights[x][i][y][j][z] = st

		if length > 4:
			print("Dimension > 4")
			sys.exit()

		store(shape)

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

print("AVG Diff")
print(avgDiff/count)
