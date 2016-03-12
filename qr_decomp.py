import random
import sys
import os
import argparse
import copy
import math


#======================================================================
# Matrix function utilities
def transpose(m):
	mT = []
	for column in range (0, len(m[0])):
		rowT = []
		for row in range (0, len(m)):
			rowT.append(m[row][column])
		mT.append(rowT)
	return mT

def remove_row(m, row):
	newMatrix = []
	for r in range(0, len(m)):
		if r != row:
			newRow = list.copy(m[r])
			newMatrix.append(newRow)
	return newMatrix

def remove_column(m, column):
	newMatrix = []
	for r in range(0, len(m)):
		newRow = []
		for c in range (0, len(m[0])):
			if c != column:
				newRow.append(m[r][c])			
		newMatrix.append(newRow)
	return newMatrix

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

def inverse(m):
	det = determinant(m) 
	if det == 0:
		print("This matrix has no inverse, returning original")
		return m
	else:
		return m_s_multiply(adjugate_matrix(m), (1 / det))


def m_s_multiply(m1, n):
	m3 = []
	for row in range (0, len(m1)):
		rowM = []
		for column in range (0, len(m1[0])):
			product = m1[row][column] * n
			rowM.append(product)
		m3.append(rowM)
	return m3

def v_s_multiply(v1, n):
	v2 = []
	for row in range (len(v1)):
		v2.append(v1[row] * n)
	return v2

def m_sum(m1, m2):
	m3 = []
	for row in range (0, len(m1)):
		row3 = []
		for column in range (0, len(m1[0])):
			row3.append(m1[row][column] + m2[row][column])
		m3.append(row3)
	return m3

def v_sum(v1, v2):
	v3 = []
	for row in range (len(v1)):
		v3.append(v1[row] + v2[row])
	return v3

def m_subtract(m1, m2):
	m3 = []
	for row in range (0, len(m1)):
		row3 = []
		for column in range (0, len(m1[0])):
			row3.append(m1[row][column] - m2[row][column])
		m3.append(row3)
	return m3

def m_add(m1, m2):
	m3 = []
	for row in range (0, len(m1)):
		row3 = []
		for column in range (0, len(m1[0])):
			row3.append(m1[row][column] + m2[row][column])
		m3.append(row3)
	return m3

def v_subtract(v1, v2):
	v3 = []
	for r in range (len(v1)):
		v3.append(v1[r] - v2[r])
	return v3

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

def m_v_multiply(m, v):
	m2 = []
	for row in range (0, len(m)):
		product = 0
		for column in range (0, len(m[0])):
			product += m[row][column] * v[column]
		m2.append(product)
	return m2

def dot_product(v1, v2):
	s = 0
	for row in range (0, len(v1)):
		s += v1[row] * v2[row]
	return s

def is_singular(m):
	return inverse(m) is not m

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

def zero_matrix(dimension):
	m = []
	for r in range (dimension):
		mRow = []
		for c in range (dimension):
			mRow.append(0)
		m.append(mRow)
	return m

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

def get_row(m, n):
	return m[n]

def get_column(m, n):
	col = []
	for r in range (len(m)):
		for c in range (len(m[0])):
			if c == n:
				col.append(m[r][c])
	return col

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
####################################################################################

def find_u(m):
	u = [[] for x in range(len(m))]
	for l in range (len(u)):
		u[l].append(m[l][0])
	for c in range (1, len(m[0])):
		mN = get_column(m, c)
		mNcopy = mN[:]
		vectorSum = [0] * len(m[0])
		for k in range (0, c):
			eC = v_normalise(get_column(u, k))
			vectorSum = v_sum(vectorSum, v_s_multiply(eC, dot_product(mNcopy, eC)))
		uC = v_subtract(mN, vectorSum)
		for r in range (len(m)):
			u[r].append(uC[r])
	return u


def qr_decomp(m):
	q = []	
	u = find_u(m)
	for i in range (len(u)):
		qRow = v_normalise(get_column(u, i))
		q.append(qRow)
	q = transpose(q)
	r = zero_matrix(len(m))
	for row in range (len(m)):
		squaredSum = 0
		for i in range (0, len(u)):
			#print("adding %f" % u[i][row])
			squaredSum += u[i][row] ** 2
		norm = math.sqrt(squaredSum)
		for c in range (row, len(m)):
			if norm != 0:
				r[row][c] = dot_product(get_column(m, c), (get_column(u, row))) / norm
			else:
				r[row][c] = 0
	return (q, r)





def qr_factor(m, qs):
	lenM = len(m)
	shiftId = m_s_multiply(identity_matrix(lenM), m[lenM - 1][lenM - 1])
	matrix = m_subtract(m, shiftId)
	(q, r) = qr_decomp(matrix)
	rq = m_m_multiply(r, q)
	ak = m_add(rq, shiftId)
	qs.append(q)
	return ak

def qr_iterative(m, threshold, qs):
	if qs == None:
		qs = []
	ak = qr_factor(m, qs)
	while (max_abs_upper_triangle(ak) > threshold):
		ak = qr_factor(ak, qs)
	return ak


def get_eigenvalues(m, threshold, qs):
	values = []
	ak = qr_iterative(m , threshold, qs)
	for r in range (len(ak)):
		for c in range (len(ak)):
			if c == r:
				eVal = ak[r][c]
				values.append(eVal)
	return values


def get_eigenvectors(m, threshold):
	qs = []
	_ = qr_iterative(m, threshold, qs)
	qk = identity_matrix(len(m))
	for i in range (len(qs)):
		qk = m_m_multiply(qk, qs[i])
	return qk


#======================================================================
# Main
def main(argv=None):
 	

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

	args = parser.parse_args(argv)
# .........................................................................
	
#..............................................................................
# Get the matrix from either a file given or generate one from the given dimensions
	matrix = []
	#matrix = Matrix([])
	dimension = int(args.n)

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

	
	# Print the matrix
	print("The Matrix given is:")
	for i in range (0,dimension):
		print(matrix[i])

	# Print QR Decomposition
	print("")
	(q, r) = qr_decomp(matrix)
	print("Q IS:")
	for i in range (len(q)):
		print(q[i])
	print("")
	print("R IS:")
	for i in range (len(r)):
		print(r[i])
	print("")

	u = find_u(matrix)
	print("U IS:")
	for i in range (len(u)):
		print(u[i])
	print("")

	# Print shifted qr iteration
	print("Final QR (calculated with shifts):")
	qs = None
	ak = qr_iterative(matrix, 0.0001, qs)
	for i in range (len(matrix)):
		print(ak[i])

	# Print eigenvalues
	eVal = get_eigenvalues(matrix, 0.0001, None)
	print("\nEigenvalues are:")
	print(eVal)

	# Print eigenvectors
	print("\nEigenvectors are:")
	qK = get_eigenvectors(matrix, 0.0001)
	for i in range (len(qK)):
		print(qK[i])


# ==============================================================================
# call main if executed as script
if __name__ == '__main__':
    sys.exit(main())


