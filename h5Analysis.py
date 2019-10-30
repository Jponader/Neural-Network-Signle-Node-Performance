
import numpy as np
import math
import sys
import h5py

if len(sys.argv) != 4:
	print("Improper Input: python3 h5Analysis.py <accrucy1> <accrucy5> <input_file>")
	sys.exit()

expected_val1 = float(sys.argv[1])
expected_val5 = float(sys.argv[2])

global_min1 = 1
global_diff1 = 0
global_count1 = 0
global_avg1 = 0
global_var1 = 0

global_min5 = 1
global_diff5 = 0
global_count5 = 0
global_avg5 = 0
global_var5 = 0

local_min = 1
local_avg = 0
local_count = 0
local_diff = 0

def tester(result, name):
	global global_min1
	global global_diff1
	global global_count1
	global global_avg1
	global global_var1

	global global_min5
	global global_diff5
	global global_count5
	global global_avg5
	global global_var5

	global local_min
	global local_avg
	global local_count
	global local_diff

		
	if("." in name):
		global_count5 = global_count5 + 1
		global_avg5 = global_avg5 + result
		global_var5 = global_var5 + ((result - expected_val5) **2)

		if result != expected_val5:
			global_diff5 = global_diff5 + 1
			local_diff = local_diff + 1

		if result < global_min5:
			global_min5 = result
	elif(name[0] != "a"):
		global_count1 = global_count1 + 1
		global_avg1 = global_avg1 + result
		global_var1 = global_var1 + ((result - expected_val1) ** 2)

		if result != expected_val1:
			global_diff1 = global_diff1 + 1
			local_diff = local_diff + 1

		if result < global_min1:
			global_min1 = result



	local_count = local_count + 1
	local_avg = local_avg + result

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
	print("----GLOBAL 1----")
	print("Global Min: " + str(global_min1))
	print("Global Count: " + str(global_count1))
	print("Global Diff: " + str(global_diff1))
	print("Global Avg: " + str(global_avg1/global_count1))
	print("Global Standard Dev: " + str(math.sqrt(global_var1/global_count1)))
	print("----GLOBAL 5----")
	print("Global Min: " + str(global_min5))
	print("Global Count: " + str(global_count5))
	print("Global Diff: " + str(global_diff5))
	print("Global Avg: " + str(global_avg5/global_count5))
	print("Global Standard Dev: " + str(math.sqrt(global_var5/global_count5)))



path_to_file = sys.argv[3]

fh5 = h5py.File(path_to_file, 'r')
for Groups in fh5:
	group = fh5[Groups]
	for layers in group:
		print("----"+layers+"----")
		shape = fh5[Groups][layers].shape
		length = len(shape)
		layer = fh5[Groups][layers]

		print(type(layers))

		if(layers[0] == "a"):
			continue

		if length == 1:
			for i in range(shape[0]):
				tester(layer[i], layers)


		if length == 2:
			for i in range(shape[0]):
				for y in range(shape[1]):
					tester(layer[i][y], layers)

		if length == 3:
			for i in range(shape[0]):
				for y in range(shape[1]):
					for j in range(shape[2]):
						tester(layer[i][y][j], layers)

		if length == 4:
			for i in range(shape[0]):
				for y in range(shape[1]):
					for j in range(shape[2]):
						for z in range(shape[3]):
							tester(layer[i][y][j][z], layers)

		if length > 4:
			print("Dimension > 4")
			sys.exit()

		printLocal()
		reset()

printGlobal()

	
