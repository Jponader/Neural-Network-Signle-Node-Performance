import numpy as np
from scipy.signal import convolve

#dense matrix solving
inputMat = np.array([[1,2,1],[3,2,4],[4,3,2]])
intermediary = np.array([[2,1,-2],[4,5,-1],[3,-4,2]])

solution = np.matmul(inputMat, intermediary)

solved = np.linalg.solve(inputMat,solution)

print("Dense Matrix")
print(inputMat)
print(intermediary)
print(solution)
print(solved)


#convolution
inputMat = np.array([[1,2,3,4],[-1,2,-3,1],[2,1,-2,-3],[4,-2,1,-4]])
inputlen = 4
intermediary = np.array([[1,0],[0,1]])
intermedLen = 2

solution = convolve(inputMat, intermediary, mode="valid")
size = len(solution)

print("Convolution")
print(inputMat)
print(intermediary)
print(solution)

solution = solution.flatten()
#solution = solution.reshape((len(solution),1))
print(solution)
interstep = np.empty((size*size,intermedLen*intermedLen),dtype=np.int8)
count = 0

for y in range(0,size):
	for x in range(0,size):
		for j in range(0,intermedLen):
			for i in range(0,intermedLen):
				if ((i == 0) and (j == 1)):
					interstep[count][2] = inputMat[y+j][x+i]
				elif((i == 1) and (j == 1)):
					interstep[count][3] = inputMat[y+j][x+i]
				else:
					interstep[count][i+j]= inputMat[y+j][x+i]
		count = count + 1



print(interstep)


solved = np.linalg.solve(interstep[:4],solution[:4])
print(solved)


