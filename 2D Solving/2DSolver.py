import numpy as np
from scipy import ndimage

from random import seed
from random import randint
from random import random
from datetime import datetime

# TODO

# Go Through Dense Layer and have everything reference M - N - P








def denseLayer(maxMatrix = 100):

	seed(datetime.now())
	M = randint(1,maxMatrix)
	N = randint(1,maxMatrix)
	P = randint(1,maxMatrix)

	raw1 = np.random.rand(M, N) 
	raw2 = np.random.rand(N, P) 

	#Dense Matrix
	print("---Dense Matrix---")

	inputMat = raw1
	weights = raw2
	inputShape = inputMat.shape
	weightShape = weights.shape

	#print("Raw Input Matrix")
	#print(inputMat)
	#print(inputShape)
	#print("Raw Weights")
	#print(weights)
	#print(weightShape)

	#assumption of 2D matrixes, 3D matrixes to be applied latter
	def denseReshape(inputMat, weights, inputShape, weightShape):
		
		M = inputShape[0]
		N =  inputShape[1]
		P = weightShape[1]	

		if N != weightShape[0]:
			print("ERPROR")
			sys.exit()

		if M < N:
			M = N
		if P < N:
			P = N

		#print(M,N,P)
		inputMat = subDenseReshape(inputMat, inputShape, M, N)
		weights = subDenseReshape(weights, weightShape, N,P)

		return inputMat, weights

	def subDenseReshape (matrix, shape, x,y):
		seed(shape[0])

		out = np.empty((x,y))
		
		for i in range(x):
			for j in range(y):
				if i < shape[0] and j < shape[1]:
					out[i][j] = matrix[i][j]
				else:
					out[i][j] = randint(0,100)
		return out

	def subDenseUnshape(matrix, shape):
		return matrix[:shape[0], :shape[1]]

	def solveDenseLayer (inputMat, weights):
		output = np.matmul(inputMat, weights)
		return output

	def denseWeightsSolver(inputMat, output):
		solvedWeights = np.linalg.lstsq(inputMat, output,rcond=-1)
		#solvedWeights = np.linalg.solve(inputMat,output)
		#solvedWeights = np.array(solvedWeights)
		return solvedWeights[0]

	def denseInputSolver(weights, output):
		#solvedInput = np.linalg.solve(weights.T, output.T)
		solvedInput = np.linalg.lstsq(weights.T, output.T,rcond=-1)
		#print(weights.T)
		#print(output.T)
		return solvedInput[0].T

	raw3= solveDenseLayer (inputMat, weights)
	outShape = raw3.shape

	inputMat,weights = denseReshape(inputMat, weights, inputShape, weightShape)

	#print("Solved Dense Output - Showing output with padding")
	output = solveDenseLayer (inputMat, weights)
	#print(output)
	#print(output.shape)

	#print("Solved Dense Weights - reshaped to proper size")
	#print(inputMat.shape)
	#print(output.shape)
	solvedWeights = denseWeightsSolver(inputMat, output)
	solvedWeights = subDenseUnshape(solvedWeights, weightShape)
	#print(solvedWeights)

	#print("Solved Dense Input - reshaped to proper size")
	#print(weights.shape)
	#print(output.shape)
	solvedInput = denseInputSolver(weights, output)
	solvedInput = subDenseUnshape(solvedInput, inputShape)
	#print(solvedInput)


	if np.allclose(solvedWeights, raw2,  atol=1e-08) != True:
		print("ERROR2 - Solved Weights")
		print(raw2)
		print(solvedWeights)
		return 2

	if np.allclose(solvedInput, raw1, atol=1e-08) != True:
		print("ERROR1 - Solved Input")
		print(raw1)
		print(solvedInput)
		return 1
	
	if np.allclose(subDenseUnshape(output, outShape), raw3,  atol=1e-08) != True:
		print("ERROR3 - Solved Output")
		print(raw3)
		print(subDenseUnshape(output, outShape))
		return 3

	return 0
#Dense Layer Testing Script

one = 0
two = 0
three = 0

for i in range(0,10):
	hold = denseLayer(maxMatrix = 100)

	if hold == 1:
		one = one + 1
	elif hold == 2:
		two = two + 1
	elif hold == 3:
		three =  three + 1

print("Summary")
print("Error 1 - Solved Input : ", one)
print("Error 2 - Solved Weights : ", two)
print("Error 3 - Solved Output : ", three)


def convolutionLayer():
	# Convolution

#with stride 1 all sizes of filters will work
# idead change M to N, so using output, random stride, and filter size calculate input size
# inputSize = ((output -1)*stride) - 2padding + filtersize
#inputlen = ((solLen - 1)*1) + filterLen

	seed(datetime.now())
	N = randint(1,15)
	F = randint(1,10)
	S = randint(1,F)
	Y = randint(1,10)

	M = ((N - 1)* S) + F

	raw1 = np.random.rand(M, M) 
	raw2 = np.random.rand(Y, F,F) 


	#N = 2
	#F = 2
	#S = 2
	#Y = 4
	#M = 4

	#raw1 = np.array([[1.0,2.0,3.0,4.0],[-1.0,2.0,-3.0,1.0],[2.0,1.0,-2.0,-3.0],[4.0,-2.0,1.0,-4.0]])
	#raw2= np.array([[[1.0,2.0],[2.0,1.0]],[[1.0,2.0],[1.0,1.0]],[[2.0,3.0],[1.0,1.0]],[[3.0,1.0],[1.0,3.0]] ])


	#print("M F Y S N")
	#print(M,F,Y,S,N)

	print("---Convolution---")
	inputMat = raw1
	filters = raw2

	#print("Input Matrix")
	#print(inputMat)
	#print(inputMat.shape)
	#print("Filters - First One Only Shown")
	#print(filters[0])
	#print(filters.shape)

	def filtersAdd (filters, countNeeded):
		shape = filters.shape
		np.random.seed(countNeeded)
		out = []

		for i in filters:
			out.append(i)

		for i in range(shape[0], countNeeded):
			out.append(np.random.rand(shape[1],shape[2]))

		return np.array(out)

	def convolutionPadding(matrix, F, S):

		M = ((F - 1)* S) + F
		shape = matrix.shape

		seed(shape[0])

		out = np.empty((M,M))
		
		for i in range(M):
			for j in range(M):
				if i < shape[0] and j < shape[1]:
					out[i][j] = matrix[i][j]
				else:
					out[i][j] = randint(0,100)
		return out

	def unshape3D(matrix, shape):
		return matrix[:shape[0],:shape[1], :shape[2]]

	def unshape2D(matrix, shape):
		return matrix[:shape[0],:shape[1]]

	def Convolution(inputMat, filter, stride):
		inputlen = len(inputMat)
		filterLen = len(filter)
		#output = ((inputSize - filterSize + 2padding)/stride )+ 1)
		interstepSize = int(((inputlen - filterLen)/stride)+1)
		solution = np.zeros((interstepSize,interstepSize))
		sub = 0
		fil = filter.flatten()

		for x in range(0,interstepSize):
			for y in range(0,interstepSize):
				for j in range(0,filterLen):
					for i in range(0,filterLen):
						sub = (fil[(j*filterLen)+i] * inputMat[(x*stride)+j][(y*stride)+i]) + sub
				solution[x][y] = sub
				sub = 0

		return solution

	def ConAllFilters(inputMat, filters, stride = 1):
		solution = []

		for filter in filters:
			solution.append(Convolution(inputMat, filter, stride))

		solution = np.array(solution)
		return solution

	def ConAllWeights(inputMat, output, stride = 1):
		solution = []

		for out in output:
			solution.append(ConWeightsSolver(inputMat, out, stride))

		solution = np.array(solution)
		return solution

	# Solving Intermediary Step
	def ConWeightsSolver(inputMat, output, stride):
		ConOut = output.flatten()
		M = len(inputMat)
		ConOutCount = len(ConOut)
		N =  len(output)
		#filter = input + 2padding -((Output -1)*stride)
		F = M -((N - 1)*stride)
		weightCount = F*F
		weights = np.zeros((ConOutCount,weightCount))
		count = 0		

		for y in range(0,N):
			for x in range(0,N):
				for j in range(0,F):
					for i in range(0,F):
						weights[count][(j*F)+i]= inputMat[(y*stride)+j][(x*stride)+i]
				count = count + 1


		solved = np.linalg.lstsq(weights,ConOut,rcond=-1)
		solved = solved[0].reshape((F,F))

		#solved = np.linalg.solve(weights[:weightCount],ConOut[:weightCount])
		#solved = solved.reshape((F,F))

		return solved

	# Solving Input 
	def SimpleConInputSolver(solution,filters, stride = 1):
		Y = len(filters)
		F = len(filters[0])
		filterSize = F*F
		weightMat = np.empty((filterSize,filterSize))
		answerMat = np.empty((filterSize))
		count = 0

		if Y < F*F:
			print("Error Not Enough Filters")

		for filter in filters:
			if count == filterSize:
				break
			flat = filter.flatten()
			for z in range(0,filterSize):
				weightMat[count][z] = flat[z]
			count = count + 1

		N = len(solution[0])
		# inputSize = ((output -1)*stride) - 2padding + filtersize
		inputlen = ((N - 1)*stride) + F
		solved = np.zeros((inputlen,inputlen))

#optimizaton formula for not double solving when filters overlap, change the stride using the filterSize and stride

		for x in range(0,N):
			for y in range(0,N):
				for z in range(0, filterSize):
					answerMat[z] = solution[z][x][y]

				intermed = np.linalg.solve(weightMat,answerMat)

				count = 0
				for i in range(0, F):
					for j in range(0, F):
						solved[(x*stride)+i][(y*stride)+j] = intermed[count]
						count = count + 1

		return solved


	raw3 = ConAllFilters(inputMat, filters, stride = S)

	if Y < (F*F):
		#print("Adding Filters", Y, F*F)
		filters = filtersAdd(filters, F*F)

	if N < F:
		print("padding")
		inputMat = convolutionPadding(inputMat, F, S, )

	#print("Solved Convolution Output - Showing Output with Padding")
	output = ConAllFilters(inputMat, filters, stride = S)
	#print(output[0])

	#print("Solved Convolution Weights")
	solvedWeights = ConAllWeights(inputMat, output,stride = S)
	#print(solvedWeights[0])

	#print("Solved Convolution Input")
	solvedInput = SimpleConInputSolver(output, filters, stride = S)
	#print(solvedInput)


	if np.allclose(solvedWeights[:Y], raw2,  atol=1e-08) != True:
		print("ERROR2 - Solved Weights")
		print(raw2)
		print(solvedWeights[:Y])
		return 2

	if np.allclose(unshape2D(solvedInput,raw1.shape), raw1, atol=1e-08) != True:
		print("ERROR1 - Solved Input")
		print(raw1)
		print(solvedInput)
		return 1

	if np.allclose(unshape3D(output,raw3.shape), raw3,  atol=1e-08) != True:
		print("ERROR3 - Solved Output")
		print(raw3)
		print(unshape3D(output,raw3.shape))
		return 3

	return 0
	
one = 0
two = 0
three = 0

for i in range(0,100):
	hold = convolutionLayer()

	if hold == 1:
		one = one + 1
	elif hold == 2:
		two = two + 1
	elif hold == 3:
		three =  three + 1

print("Summary")
print("Error 1 - Solved Input : ", one)
print("Error 2 - Solved Weights : ", two)
print("Error 3 - Solved Output : ", three)
