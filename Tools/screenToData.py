
import numpy as np
import sys
import h5py

if len(sys.argv) != 3:
	print("Improper Input: python3 screenToData.py <input_file> <output_location>")
	sys.exit()

input_file = sys.argv[1]
output = sys.argv[2]

fh5 = h5py.File(output, 'w')
fh5 = fh5.create_group("Group")

dataStore = []
layer = 0

fp =open(input_file, 'r')

line = fp.readline()

while line:
	if line[0] == '7':
		line = fp.readline()
		continue
	if line[0] == '-':
		fh5.create_dataset(str(layer), data=dataStore)
		dataStore = []
		layer = layer + 1
		line = fp.readline()
		continue

	if line[0] == 'c':
		print(line)
		d_line = float(line[3:-1])
		dataStore.append(d_line)
		line = fp.readline()
		continue

	line = fp.readline()
