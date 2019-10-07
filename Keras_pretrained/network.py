from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

from keras import optimizers
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
#from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
#rom keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
#from keras.applications.nasnet import NASNetMobile, preprocess_input, decode_predictions
from keras.preprocessing.image import load_img, img_to_array
from keras.utils import to_categorical, HDF5Matrix

# Helper libraries
import numpy as np
from numpy import ndarray as nd

import array as array

import matplotlib.pyplot as plt
import scipy.io

import time
import pathlib
import random
import h5py

test_length = 1000


model = ResNet50(weights='imagenet')
#model = VGG16(weights='imagenet')
#model = MobileNetV2(weights='imagenet')
#model = NASNetMobile(weights='imagenet')

keras.backend.set_learning_phase(0)

# Save Weights
#model.save_weights('weights.h5')
# model.load_weights()

# Save Entire Model
# model.save('model.h5')
# model = keras.models.load_model('model.h5')

path_val = '../ImageNet/ILSVRC/Data/CLS-LOC/val'
label_text = '../ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt'

data_root = pathlib.Path(path_val)

# use */* for folders containig images
all_image_paths = list(data_root.glob('*'))
all_image_paths = [str(path) for path in all_image_paths]
all_image_paths.sort()

"""
hf = h5py.File("save_data/imgs.h5", "w")
imgs = hf.create_dataset("head", (50000,224,224,3), dtype=np.uint8)
i = 0

for i,paths in enumerate(all_image_paths):
	print(paths)
	original_image = load_img(paths, target_size=(224, 224)) 
	numpy_image = img_to_array(original_image)
	input_image = np.expand_dims(numpy_image, axis=0)
	input_image = preprocess_input(input_image)
	imgs[i] = input_image[0]

hf.close()
"""
#imgs = HDF5Matrix("save_data/imgs.h5", "head")

#imgs = imgs[:1000]
#imgs = np.array(imgs)

imgs = []
i = 1

for paths in all_image_paths:
	original_image = load_img(paths, target_size=(224, 224)) 
	print(paths)
	numpy_image = img_to_array(original_image)
	input_image = np.expand_dims(numpy_image, axis=0) 
	input_image = preprocess_input(input_image)
	imgs.append(input_image[0])
	if i >= test_length:
		break
	i = i + 1

imgs = np.array(imgs)

"""
#Single Image Classification
img_path = '../ImageNet/ILSVRC/Data/CLS-LOC/val/ILSVRC2012_val_00000003.JPEG'
img = load_img(img_path, target_size=(224, 224))
x = img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
print(preds)
print(preds.argmax())
print('Predicted:', decode_predictions(preds, top=3)[0])
"""

# Label Matching From 
# https://calebrob.com/ml/imagenet/ilsvrc2012/2018/10/22/imagenet-benchmarking.html
# https://github.com/calebrob6/imagenet_validation
meta = scipy.io.loadmat("../ILSVRC2012_devkit_t12/data/meta.mat")
original_idx_to_synset = {}
synset_to_name = {}

for i in range(1000):
    ilsvrc2012_id = int(meta["synsets"][i,0][0][0][0])
    synset = meta["synsets"][i,0][1][0]
    name = meta["synsets"][i,0][2][0]
    original_idx_to_synset[ilsvrc2012_id] = synset
    synset_to_name[synset] = name

synset_to_keras_idx = {}
keras_idx_to_name = {}
f = open("../synset_words.txt","r")
idx = 0
for line in f:
    parts = line.split(" ")
    synset_to_keras_idx[parts[0]] = idx
    keras_idx_to_name[idx] = " ".join(parts[1:])
    idx += 1
f.close()

def convert_original_idx_to_keras_idx(idx):
    return synset_to_keras_idx[original_idx_to_synset[idx]]

f = open(label_text,"r")
labels = f.read().strip().split("\n")
labels = list(map(int, labels))
labels = np.array([convert_original_idx_to_keras_idx(idx) for idx in labels])
labels = to_categorical(labels, 1000)
f.close()


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

