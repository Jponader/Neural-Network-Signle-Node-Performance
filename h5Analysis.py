
import numpy as np
import sys
import h5py


global_min = 1
global_diff = 0
global_count = 0
global_avg = 0

local_min = 1
local_avg = 0
local_count = 0
local_diff = 0

def tester(result):
	global global_min
	global global_diff
	global global_count
	global global_avg

	global local_min
	global local_avg
	global local_count
	global local_diff

	global_count = global_count + 1
	local_count = local_count + 1

	global_avg = global_avg + result
	local_avg = local_avg + result

	if result != 1:
		global_diff = global_diff + 1
		local_diff = local_diff + 1

	if result < global_min:
		global_min = result

	if result < local_min:
		local_min = result


def reset ():
	global local_min
	global local_avg
	global local_count
	global local_diff

	local_min = 1
	local_avg = 0.0
	local_count = 0
	local_diff = 0



def printLocal():
	print("Local Min: " + str(local_min))
	print("Local Count: " + str(local_count))
	print("Local Diff: " + str(local_diff))
	print("Local Avg: " + str(local_avg/local_count))

def printGlobal():
	print("----GLOBAL----")
	print("Global Min: " + str(global_min))
	print("Global Count: " + str(global_count))
	print("Global Diff: " + str(global_diff))
	print("Global Avg: " + str(global_avg/global_count))

if len(sys.argv) != 2:
	print("Improper Input: python3 h5Analysis.py <input_file>")
	sys.exit()

path_to_file = sys.argv[1]

fh5 = h5py.File(path_to_file, 'r')
for Groups in fh5:
	group = fh5[Groups]
	for layers in group:
		print("----"+layers+"----")
		shape = fh5[Groups][layers].shape
		length = len(shape)
		layer = fh5[Groups][layers]

		if length == 1:
			for i in range(shape[0]):
				tester(layer[i])


		if length == 2:
			for i in range(shape[0]):
				for y in range(shape[1]):
					tester(layer[i][y])

		if length == 3:
			for i in range(shape[0]):
				for y in range(shape[1]):
					for j in range(shape[2]):
						tester(layer[i][y][j])

		if length == 4:
			for i in range(shape[0]):
				for y in range(shape[1]):
					for j in range(shape[2]):
						for z in range(shape[3]):
							tester(layer[i][y][j][z])

		if length > 4:
			print("Dimension > 4")
			sys.exit()

		printLocal()
		reset()

printGlobal()

	