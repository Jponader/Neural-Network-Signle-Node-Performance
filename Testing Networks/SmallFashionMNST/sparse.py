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

#-------------------------------------------------------------------------------#

global_pred = model.predict(imgs[:test_length], verbose = 1)
global_acc1 = top_k_accuracy(labels[:test_length], global_pred,1)
global_acc5 = top_k_accuracy(labels[:test_length], global_pred,5)

print(global_acc1)
print(global_acc5)

changed = 0
trueSparse = 0

val = .25
negval = val * -1


# managing weights
model.summary()
rowcount = 0;

q = 0
for layer in model.layers:

	weights = layer.get_weights()
	weights = np.array(weights)

	for x in range(len(weights)):

		print("----------------BREAK----------------")
		shape = weights[x].shape
		size = weights[x].size
		length = len(shape)

		if length == 0:
			continue

		if length == 1:
			for i in range(shape[0]):

				if weights[x][i] < val and weights[x][i] > negval:
					weights[x][i] = 0
					changed = changed + 1

				elif weights[x][i] == 0 :
					changed = changed + 1
					trueSparse = trueSparse + 1


			layer.set_weights(weights)
		

		if length == 2:
			for i in range(shape[0]):
				for y in range(shape[1]):
					
					if weights[x][i][y] < val and weights[x][i][y] > negval:
						weights[x][i][y] = 0
						changed = changed + 1

					elif weights[x][i][y] == 0 :
						changed = changed + 1
						trueSparse = trueSparse + 1


			layer.set_weights(weights)


		if length == 3:
			for i in range(shape[0]):
				for y in range(shape[1]):
					for j in range(shape[2]):
						if weights[x][i][y][j] < val and weights[x][i][y][j] > negval:
							weights[x][i][y][j] = 0
							changed = changed + 1

						elif weights[x][i][y][j] == 0 :
							changed = changed + 1
							trueSparse = trueSparse + 1

			layer.set_weights(weights)

		if length == 4:
			for i in range(shape[0]):
				for y in range(shape[1]):
					for j in range(shape[2]):
						for z in range(shape[3]):
							if weights[x][i][y][j][z] < val and weights[x][i][y][j][z] > negval:
								weights[x][i][y][j][z] = 0
								changed = changed + 1

							elif weights[x][i][y][j][z] == 0 :
								changed = changed + 1
								trueSparse = trueSparse + 1

			layer.set_weights(weights)

		if length > 4:
			print("Dimension > 4")
			sys.exit()


model.save('sparse.h5')

global_pred = model.predict(imgs[:test_length], verbose = 1)

global_acc1 = top_k_accuracy(labels[:test_length], global_pred,1)
global_acc5 = top_k_accuracy(labels[:test_length], global_pred,5)

print(global_acc1)
print(global_acc5)
print(trueSparse)
print(changed)