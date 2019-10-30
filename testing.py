import numpy as np
from scipy import ndimage

#Dense Matrix
print("---Dense Matrix---")
inputMat = np.array([[1,2,3,4],[7,9,10,11],[6,5,4,6]])
weights = np.array([[7],[5],[3],[1]])

print(inputMat)
print(weights)

def solveDenseLayer (inputMat, weights):
	output = np.matmul(inputMat, weights)
	return output

def denseWeightsSolver(inputMat, output):
	solvedWeights = np.linalg.lstsq(inputMat,output, rcond=-1)
	#solvedWeights = np.array(solvedWeights)
	return solvedWeights[0]

def denseInputSolver(weights, output):
	solvedInput = np.linalg.lstsq(weights.T, output.T,rcond=-1)
	print(weights.T)
	print(output.T)
	return solvedInput[0].T


print("Solved Dense Output")
output = solveDenseLayer (inputMat, weights)
print(output)

print("Solved Dense Weights")
solvedWeights = denseWeightsSolver(inputMat, output)
print(solvedWeights)

print("Solved Dense Input")
solvedInput = denseInputSolver(weights, output)
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
	#output = ((inputSize - filterSize + 2padding)/stride )+ 1)
	interstepSize = int(((inputlen - filterLen)/1)+1)
	solution = np.zeros((interstepSize,interstepSize))
	sub = 0
	fil = filter.flatten()

	for x in range(0,interstepSize):
		for y in range(0,interstepSize):
			for j in range(0,filterLen):
				for i in range(0,filterLen):
					sub = (fil[(j*filterLen)+i] * inputMat[x+j][y+i]) + sub
			solution[x][y] = sub
			sub = 0

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
	inputlen = len(inputMat)
	size = len(solution)
	passes =  len(output)
	#filter = input + 2padding -((Output -1)*stride)
	filterLen = inputlen -((passes - 1)*1)
	interstep = np.zeros((size,filterLen*filterLen),dtype=np.int8)
	count = 0

	for y in range(0,passes):
		for x in range(0,passes):
			for j in range(0,filterLen):
				for i in range(0,filterLen):
						interstep[count][(j*filterLen)+i]= inputMat[y+j][x+i]
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