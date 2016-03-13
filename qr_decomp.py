import random
import sys
import os
import argparse
import copy
import math


#======================================================================
###############   MATRIX UTILITIES    #################################
#======================================================================


# Returns the transpose of a matrix
def transpose(m):
	mT = []
	for column in range (0, len(m[0])):
		rowT = []
		for row in range (0, len(m)):
			rowT.append(m[row][column])
		mT.append(rowT)
	return mT

# Returns a new (not modifying m) matrix without column specified
def remove_row(m, row):
	newMatrix = []
	for r in range(0, len(m)):
		if r != row:
			newRow = list.copy(m[r])
			newMatrix.append(newRow)
	return newMatrix

# Returns a new (not modifying m) matrix without column specified
def remove_column(m, column):
	newMatrix = []
	for r in range(0, len(m)):
		newRow = []
		for c in range (0, len(m[0])):
			if c != column:
				newRow.append(m[r][c])			
		newMatrix.append(newRow)
	return newMatrix

# Finds the determinant of a matrix
def determinant(m):
	if len(m) == 2:
		return ((m[0][0] * m[1][1]) - (m[0][1] * m[1][0]))
	else:
		sum = 0
		for column in range (0, len(m[0])):
			num = m[0][column]
			newDet1 = remove_row(m, 0)
			newDet = remove_column(newDet1, column)
			if (column + 1) % 2 == 0:
				sum +=  determinant(newDet) * num
			else:
				sum -= determinant(newDet) * num
		return sum

# Find the inverse of a matrix, returns itself if determinant is 0
def inverse(m):
	det = determinant(m) 
	if det == 0:
		print("This matrix has no inverse, returning original")
		return m
	else:
		return m_s_multiply(adjugate_matrix(m), (1 / det))


# Scalar matrix multiplication
def m_s_multiply(m1, n):
	m3 = []
	for row in range (0, len(m1)):
		rowM = []
		for column in range (0, len(m1[0])):
			product = m1[row][column] * n
			rowM.append(product)
		m3.append(rowM)
	return m3

# Scalar multiplication of vector
def v_s_multiply(v1, n):
	v2 = []
	for row in range (len(v1)):
		v2.append(v1[row] * n)
	return v2


# Adds two matrices
def m_sum(m1, m2):
	m3 = []
	for row in range (0, len(m1)):
		row3 = []
		for column in range (0, len(m1[0])):
			row3.append(m1[row][column] + m2[row][column])
		m3.append(row3)
	return m3

# Adds two vectors
def v_sum(v1, v2):
	v3 = []
	for row in range (len(v1)):
		v3.append(v1[row] + v2[row])
	return v3

# Subtracts two matrices
def m_subtract(m1, m2):
	m3 = []
	for row in range (0, len(m1)):
		row3 = []
		for column in range (0, len(m1[0])):
			row3.append(m1[row][column] - m2[row][column])
		m3.append(row3)
	return m3

# Adds matrix m1 to matrix m2
def m_add(m1, m2):
	m3 = []
	for row in range (0, len(m1)):
		row3 = []
		for column in range (0, len(m1[0])):
			row3.append(m1[row][column] + m2[row][column])
		m3.append(row3)
	return m3

# Subtracts v2 from v1
def v_subtract(v1, v2):
	v3 = []
	for r in range (len(v1)):
		v3.append(v1[r] - v2[r])
	return v3

# Multiplies a matrix with a matrix (must be a well defined operation)
def m_m_multiply(m1, m2):
	m3 = []
	for row in range (0, len(m1)):
		rowM = []
		for column in range (0, len(m2)):
			product = 0
			for c in range(0, len(m1[0])):
				product += m1[row][c] * m2[c][column]
			rowM.append(product)
		m3.append(rowM)
	return m3

# Multiplies a matrix with a vector
def m_v_multiply(m, v):
	m2 = []
	for row in range (0, len(m)):
		product = 0
		for column in range (0, len(m[0])):
			product += m[row][column] * v[column]
		m2.append(product)
	return m2

# Dot product of 2 vectors
def dot_product(v1, v2):
	s = 0
	for row in range (0, len(v1)):
		s += v1[row] * v2[row]
	return s

# Is a matrix singular?
def is_singular(m):
	return inverse(m) is not m

# Adjugate matrix used in calculating the determinant
def adjugate_matrix(m):
	cofactor = []
	for r in range (0, len(m)):
		newRow = []
		for c in range (0, len(m[0])):
			new1 = remove_row(m, r)
			new2 = remove_column(new1, c)
			if (c + r + 1) % 2 == 0:
				newRow.append(determinant(new2))
			else:
				newRow.append(determinant(new2) * -1)
		cofactor.append(newRow)
	return transpose(cofactor)


# Normalises a vector, returns itself if norm is 0
def v_normalise(v):
	newV = []
	squaredSum = 0
	for i in range (0, len(v)):
		squaredSum += v[i] ** 2
	norm = math.sqrt(squaredSum)
	if norm == 0:
		return v
	for j in range (0, len(v)):
		newV.append(v[j] / norm)
	return newV

# Returns a normalised matrix, itself if the determinant is 0
def m_normalise(m):
	normQ = []
	det = determinant(m)
	if det == 0:
		print("this matrix has det 0")
		return m
	for r in range (len(m)):
		normRow = []
		for c in range (len(m[0])):
			normRow.append(m[r][c] / det)
		normQ.append(normRow)
	return normQ

# Zero matrix size n
def zero_matrix(dimension):
	m = []
	for r in range (dimension):
		mRow = []
		for c in range (dimension):
			mRow.append(float(0))
		m.append(mRow)
	return m

# Zero vector size n
def zero_vector(dimension):
	m = []
	for c in range (dimension):
		m.append(float(0))
	return m

# Identity matrix size n
def identity_matrix(n):
	m = []
	for r in range (n):
		mRow = []
		for c in range (n):
			if c == r:
				mRow.append(1)
			else:
				mRow.append(0)
		m.append(mRow)
	return m

# Get nth row of matrix m
def get_row(m, n):
	return m[n]

# Get column n of matrix m
def get_column(m, n):
	col = []
	for r in range (len(m)):
		for c in range (len(m[0])):
			if c == n:
				col.append(m[r][c])
	return col

# Returns the largest absolute value of the upper triangle of a matrix (used for threshold checking)
def max_abs_upper_triangle(m):
	utMax = abs(m[0][1])
	for r in range (len(m)):
		for c in range (r+1, len(m)):
			mRC = m[r][c]
			if abs(mRC) > utMax:
				utMax = abs(mRC)
	return utMax


# Random symmetric matrix generator. Generates random floats in the range -100000, 100000
def create_symmetric_matrix(dimension):
	matrix = []
	columns_covered = 0
	for r in range (0, dimension):
		row = []
		for c in range (0, dimension):
			if columns_covered > 0 and c < columns_covered:
				row.append(matrix[c][r])
			else:
				row.append(random.random() * random.randint(-100000, 100000))
		matrix.append(row)
		columns_covered += 1

	return matrix



#======================================================================
###############   QR FUNCTIONALITY    #################################
#======================================================================

# Finds the U matrix used in QR decomposition
def find_u(m):
	# Append original Matrix elements to the row
	u = [[] for x in range(len(m))]
	for l in range (len(u)):
		u[l].append(m[l][0])

	# Apply Gram Schmidt to calculate the sum to subtract
	for c in range (1, len(m[0])):
		mN = get_column(m, c)
		mNcopy = mN[:]
		vectorSum = [0] * len(m[0])
		for k in range (0, c):
			eC = v_normalise(get_column(u, k))
			vectorSum = v_sum(vectorSum, v_s_multiply(eC, dot_product(mNcopy, eC)))

		# Subtract the vector sum
		uC = v_subtract(mN, vectorSum)
		for r in range (len(m)):
			u[r].append(uC[r])
	return u

# Find Q for QR decomposition
def find_q(m):
	q = []
	u = find_u(m)
	for i in range (len(u)):
		# Normalise each column of u and append to q
		qRow = v_normalise(get_column(u, i))
		q.append(qRow)
	return transpose(q)

# Find R for QR decomposition
def find_r(m, q):
	# R = Q^T . M
	return m_m_multiply(transpose(q), m)

# Return pair Q and R
def qr_decomp(m):
	q = find_q(m)
	r = find_r(m, q)
	return (q, r)


# QR iteration with shifts
def qr_factor(m, qs):
	lenM = len(m)
	# Last element * Id matrx
	shiftId = m_s_multiply(identity_matrix(lenM), m[lenM-1][lenM-1])
	matrix = m_subtract(m, shiftId)
	# Decompose the matrix
	(q, r) = qr_decomp(matrix)
	rq = m_m_multiply(r, q)
	# Add the shift back on
	ak = m_add(rq, shiftId)
	# Append the q onto the qs list
	qs.append(q)
	return ak

# Iteratively calculate qr until it's within the threshold of acceptance
def qr_iterative(m, threshold, qs):
	if qs == None:
		qs = []
	ak = qr_factor(m, qs)
	while (max_abs_upper_triangle(ak) > threshold):
		ak = qr_factor(ak, qs)
	return (ak, qs)

# Retrieve eigenvalues from m (recalculates everything)
def get_raw_eigenvalues(m, threshold, qs):
	values = []
	ak, _ = qr_iterative(m , threshold, qs)
	for r in range (len(ak)):
		for c in range (len(ak)):
			if c == r:
				# Obtain diagonal elements of final matrix
				eVal = ak[r][c]
				values.append(eVal)
	return values

# Retrieve eigenvectors from m (recalculates everything)
def get_raw_eigenvectors(m, threshold):
	qs = []
	_, qList = qr_iterative(m, threshold, qs)
	qk = identity_matrix(len(m))
	# Multiply each of the Q's in the qs list to get the eigenvectors
	for i in range (len(qList)):
		qk = m_m_multiply(qk, qs[i])
	return qk

# Retrieve eigenvalues from a previously calculated qr iteration
def get_eigenvalues(ak):
	values = []
	for r in range (len(ak)):
		for c in range (len(ak)):
			if c == r:
				# Obtain diagonal elements of final matrix
				eVal = ak[r][c]
				values.append(eVal)
	return values

# Retrieve eigenvectors from a previously calculated qr iteration (takes the qs list)
def get_eigenvectors(qs):
	qk = identity_matrix(len(qs[0][0]))
	# Multiply each of the Q's in the qs list to get the eigenvectors
	for i in range (len(qs)):
		qk = m_m_multiply(qk, qs[i])
	return qk



#======================================================================
###############   Main    #################################
#======================================================================
def main(argv=None):


	defaultThreshold = 0.001
	fileName = "qrOutput"
# ..........................................................................
  # parse command line arguments (argparse)
	parser = argparse.ArgumentParser()

	parser.add_argument('--n',
						help='The dimension to give the matrix if not given from a file',
						required=True
						)
	
	parser.add_argument('--fileGiven',
						help='Call this argument if input given from file',
						action='store_true',
						default= False,
						required=False)

	parser.add_argument('--filePath',
						help='The input file where a matrix is taken from',
						required = False
						)
	parser.add_argument('--threshold',
						help="The accuracy threshold for which the matrix should approximate 0 (0.001 by default)",
						default = defaultThreshold,
						required = False
						)
	parser.add_argument('--write',
						help='Call this argument to write output to a file called "qrOutput" ',
						action='store_true',
						default= False,
						required=False)

	args = parser.parse_args(argv)
# .........................................................................
	
#..............................................................................
# Get the matrix from either a file given or generate one from the given dimensions
	matrix = []
	dimension = int(args.n)

	# If write has been given, it will write the output to an output file
	write = args.write
	newFile = []

	# Parse numbers into the matrix if file given
	if args.fileGiven:
		sourceFile = open(args.filePath, 'r')
		for line in sourceFile:
			numberStrs = line.split()
			nums = [float(x) for x in numberStrs]
			matrix.append(nums)		
		sourceFile.close()
	# Create our own random symmetric matrix given a dimension	
	else:
		matrix = create_symmetric_matrix(dimension)

	givenThreshold = args.threshold 

	# Print the matrix
	mString = "The matrix given is:" 
	print(mString)
	newFile.append(mString)
	for i in range (0,dimension):
		print(matrix[i])
		newFile.append(str(matrix[i]))
		

	# Print QR Decomposition
	(q, r) = qr_decomp(matrix)
	
	qString = "\nQ IS:"
	newFile.append(qString)
	print(qString)
	for i in range (len(q)):
		print(q[i])
		newFile.append(str(q[i]))
	
	rString = "\nR IS:"
	print(rString)
	newFile.append(rString)
	for i in range (len(r)):
		print(r[i])
		newFile.append(str(r[i]))
	

	# Print shifted qr iteration
	qrString = "\nFinal QR iteration (calculated with shifts):"
	print(qrString)
	newFile.append(qrString)
	qs = None
	ak, qList = qr_iterative(matrix, float(givenThreshold), qs)
	for i in range (len(matrix)):
		print(ak[i])
		newFile.append(str(ak[i]))

	# Print eigenvalues (no recalculation)
	eVal = get_eigenvalues(ak)
	eValString = "\nEigenvalues are:"
	print(eValString)
	newFile.append(eValString)
	print(eVal)
	newFile.append(str(eVal))

	# Print eigenvectors (no recalculation)
	eVecString = "\nEigenvectors are:"
	print(eVecString)
	newFile.append(eVecString)
	qK = get_eigenvectors(qList)
	for i in range (len(qK)):
		print(qK[i])
		newFile.append(str(qK[i]))


	if write:
		outputFile = open('qrOutput', 'w')
		for i in range (len(newFile)):
			outputFile.write(newFile[i])
			outputFile.write('\n')


# ==============================================================================
# call main if executed as script
if __name__ == '__main__':
    sys.exit(main())


