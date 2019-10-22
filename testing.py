import numpy as np
from scipy.signal import convolve


	

#Dense Matrix
inputMat = np.array([[1,2,1],[3,2,4],[4,3,2]])
weights = np.array([[2,1,-2],[4,5,-1],[3,-4,2]])

output = np.matmul(inputMat, weights)


solvedWeights = np.linalg.solve(inputMat,output)
print(weights)
print(weights.T)

solvedInput = np.linalg.solve(weights.T, output)





print("---Dense Matrix---")
print(inputMat)
print(weights)
print("Solved Dense Output")
print(output)
print("Solved Dense Weights")
print(solvedWeights)
print("Solved Dense Input")
print(solvedInput)


# Convolution
print("---Convolution---")
inputMat = np.array([[1,2,3,4],[-1,2,-3,1],[2,1,-2,-3],[4,-2,1,-4]])
filters= np.array([	[[1,2],[2,1]],
					[[1,2],[1,1]],
					[[2,3],[1,1]],
					[[3,1],[1,3]] ])

print(inputMat)
print(filters[0])

def Convolution(inputMat, filter):
	inputlen = len(inputMat)
	filterLen = len(filter)
	#((inputSize - filterSize + 2padding)/stride )+ 1)
	interstepSize = int(((inputlen - filterLen)/1)+1)
	solution = np.zeros((interstepSize,interstepSize))

	for x in range(0,interstepSize):
		for y in range(0,interstepSize):
			sub0= filter[0][0] * inputMat[x][y]
			sub1= filter[0][1] * inputMat[x][y+1]
			sub2= filter[1][0] * inputMat[x+1][y]
			sub3= filter[1][1] * inputMat[x+1][y+1]
			solution[x][y] = sub0 + sub1 + sub2 + sub3

	return solution	

def ConAllFilters(inputMat, filters):
	solution = []

	for filter in filters:
		solution.append(Convolution(inputMat, filter))

	solution = np.array(solution)
	return solution


# Solving Intermediary Step
def ConWeightsSolver(inputMat, output):
	solution = output.flatten()
	filterLen = len(filters[0])
	inputlen = len(inputMat)
	size = len(output)
	interstep = np.empty((size*size,filterLen*filterLen),dtype=np.int8)
	count = 0

	for y in range(0,size):
		for x in range(0,size):
			for j in range(0,filterLen):
				for i in range(0,filterLen):
					if ((i == 0) and (j == 1)):
						interstep[count][2] = inputMat[y+j][x+i]
					elif((i == 1) and (j == 1)):
						interstep[count][3] = inputMat[y+j][x+i]
					else:
						interstep[count][i+j]= inputMat[y+j][x+i]
			count = count + 1

	solved = np.linalg.solve(interstep[:inputlen],solution[:inputlen])
	solved = solved.reshape((2,2))
	return solved

# Solving Input 
def SimpleConInputSolver(solution,filters):
	#stride size only 1

	filterLen = len(filters[0])
	filterVar = filterLen*filterLen
	weightMat = np.empty((filterVar,filterVar),dtype=np.int8)
	answerMat = np.empty((filterVar),dtype=np.int8)
	count = 0

	if len(filters) < filterVar:
		print("error: Not enough filters to Solve")
		return

	for filter in filters:
		if count == filterVar:
			break
		flat = filter.flatten()
		for z in range(0, filterVar):
			weightMat[count][z] = flat[z]
		count = count + 1

	solLen = len(solution[0])
	# inputSize = ((output -1)*stride) - 2padding + filtersize
	inputlen = ((solLen - 1)*1) + filterLen
	solved = np.zeros((inputlen,inputlen))

	for x in range(0,solLen, 2):
		for y in range(0,solLen, 2):
			for z in range(0, filterVar):
				answerMat[z] = solution[z][x][y]

			intermed = np.linalg.solve(weightMat,answerMat)

			count = 0
			for i in range(0, filterLen):
				for j in range(0, filterLen):
					solved[x+i][y+j] = intermed[count]
					count = count + 1

	return solved

print("Solved Convolution Output")
output = ConAllFilters(inputMat, filters)

for outputs in output:
	print(outputs)

print("Solved Convolution Weights")
filter0 = ConWeightsSolver(inputMat, output[0])
print(filter0)

print("Solved Convolution Input")
hold = SimpleConInputSolver(output, filters)
print(hold)




"""
#(inputSize - filterSize + 2padding)/(stride + 1)
interstepSize = int(((inputlen - filterLen)/1)+1)
interstep = np.zeros((interstepSize*interstepSize*2,inputlen*inputlen),dtype=np.int8)
interstepSol = np.empty((interstepSize*interstepSize*2),dtype=np.int8)
output = output.flatten
count = 0

"""
"""
for i in range (0, interstepSize*interstepSize*2):
	for j in range ((inputlen*inputlen), (interstepSize*interstepSize*2)):
		interstep[i][j] = 1
"""
"""
for y in range(0,interstepSize):
	for x in range(0,interstepSize):
		for j in range(0,filterLen):
			for i in range(0,filterLen):
				interstep[count][((y+j)*4)+x+i]= filter1[j][i]
		count = count + 1

for y in range(0,interstepSize):
	for x in range(0,interstepSize):
		for j in range(0,filterLen):
			for i in range(0,filterLen):
				interstep[count][((y+j)*4)+x+i]= filter2[j][i]
		count = count + 1

print(interstep[:inputlen*inputlen])
count = 0
solLen = len(solution)

for i in range(0,solLen):
	interstepSol [count] = solution[i]
	count = count + 1

for i in range(0,solLen):
	interstepSol [count] = solution2[i]
	count = count + 1

print(interstepSol)


#solved = solved.reshape((4,4))
#inverse = np.linalg.inv(interstep)

solved = np.linalg.solve(interstep[:inputlen*inputlen],interstepSol[:inputlen*inputlen])
print(solved)
print((np.dot(interstep,solved)))
print(np.allclose(np.dot(interstep,solved),interstepSol))
"""