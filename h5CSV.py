
import numpy as np
import sys
import h5py

if len(sys.argv) != 3:
	print("Improper Input: python3 h5CSV.py <input_file> <output_location>")
	sys.exit()

path_to_file = sys.argv[1]

fh5 = h5py.File(path_to_file, 'r')

for layers in fh5:
	shape = fh5[layers].shape
	length = len(shape)
	layer = fh5[layers]
	file = open((sys.argv[2]+"_"+str(layers)+".csv"), "w")
	print(shape)

	if length == 1:
		for i in range(shape[0]):
			print(layer[i], end = ',', file=file)


	if length == 2:
		for i in range(shape[0]):
			for y in range(shape[1]):
				print(layer[i][y], end = ',', file=file)
			print(file=file)

	if length > 2:
		print("Dimension > 2")
		sys.exit()

	if length == 3:
		for i in range(shape[0]):
			for y in range(shape[1]):
				for j in range(shape[2]):
					print(layer[i][y][j], end = ',', file=file)

	if length == 4:
		for i in range(shape[0]):
			for y in range(shape[1]):
				for j in range(shape[2]):
					for z in range(shape[3]):
						print(layer[i][y][j][z], end = ',', file=file)


	if length > 4:
		print("Dimension > 4")
		sys.exit()
	