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
		return s_multiply(adjugate_matrix(m), (1 / det))


def m_s_multiply(m1, n):
	m3 = []
	for row in range (0, len(m1)):
		rowM = []
		for column in range (0, len(m1[0])):
			product = m1[row][column] * n
			product = round(product, 10)
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
				product = round(product, 8)
			rowM.append(product)
		m3.append(rowM)
	return m3


def m_v_multiply(m, v):
	m2 = []
	for row in range (0, len(m)):
		product = 0
		for column in range (0, len(m[0])):
			#for c in range(0, len(m[0])):
			product += m[row][column] * v[column]

			product = round(product, 8)
			#rowM.append(product)
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

def normalise(v):
	newV = []
	squaredSum = 0
	for i in range (0, len(v)):
		squaredSum += v[i] ** 2
	norm = math.sqrt(squaredSum)
	for j in range (0, len(v)):
		newV.append(v[j] / norm)
	return newV



#======================================================================
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




def qr_decomp(m, q, r):
	q = find_q(m)
	r = find_r(m, q)
	return

def find_r(m, q):
	return m_m_multiply(transpose(find_q(m)), m)

def find_q(m):
	q = []
	u = find_u(m)
	for i in range (len(u)):
		qRow = normalise(get_column(u, i))
		q.append(qRow)
	return transpose(q)

def find_u(m):
	u = [[] for x in range(len(m))]
	for l in range (len(u)):
		u[l].append(m[l][0])

	for c in range (1, len(m[0])):
		mN = get_column(m, c)
		mNcopy = mN[:]

		vectorSum = [0] * len(m[0])
		for k in range (0, c):
			eC = normalise(get_column(u, k))

			vectorSum = v_sum(vectorSum, v_s_multiply(eC, dot_product(mNcopy, eC)))

		uC = v_subtract(mN, vectorSum)

		for r in range (len(m)):
			u[r].append(uC[r])
	return u


def get_row(m, n):
	return m[n]

def get_column(m, n):
	col = []
	for r in range (len(m)):
		for c in range (len(m[0])):
			if c == n:
				col.append(m[r][c])
	return col


def power_it(matrix, init_v, threshold, prevRatio):
	xk = m_v_multiply(matrix, init_v)
	ratio = xk[0] / init_v[0]
	if (prevRatio != None and abs(ratio - prevRatio) < threshold):
		return ratio
	else:
		return power_it(matrix, xk, threshold, ratio)


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
			nums = [int(x) for x in numberStrs]
			matrix.append(nums)		
		sourceFile.close()

	# Create our own random symmetric matrix given a dimension	
	else:
		matrix = create_symmetric_matrix(dimension)

	
	# Print the matrix
	for i in range (0,dimension):
		print(matrix[i])

	
	# Print determinant of matrix
	# t = determinant(matrix)
	# print("")
	# print(t)


	# Print transpose of matrix
	# t = transpose(matrix)
	# print("Transpose of matrix is:")
	# for i in range (0,dimension):
	# 	print(t[i])

	# Print inverse of matrix
	# inv = inverse(matrix)
	# print("Inverse of matrix is:")
	# for i in range (0,dimension):
	# 	print(inv[i])

	# Print scalar multiplication of matrix
	# s = s_multiply(matrix, 2)
	# print("Scalar of matrix is:")
	# for i in range (0,dimension):
	# 	print(s[i])


	# adjugate = adjugate_matrix(matrix)
	# print("adjugate Matrix is:")
	# for i in range (0, dimension):
	# 	print(adjugate[i])

	# Print multiplication of matrices
	# v1 = [-6, 8]
	# v2 = [5, 12]
	# m3 = m_m_multiply(v1, v2)
	# print("Matrix times inverse  is ")
	# for i in range (0, len(m3)):
	# 	print(m3[i])

	# Print dot product of vectors
	# v1 = [-6, 8]
	# v2 = [5, 12]
	# v3 = dot_product(v1, v2)
	# print("Dot product of v1 and 2 is")
	# print(v3)

	# Print determinant of matrix 
	# d = determinant(matrix)
	# print("")
	# print(d)

	# Print removing a row from a matrix 
	# n = remove_row(matrix, 1)
	# print("")
	# for i in range (0, dimension-1):
	# 	print(n[i])

	# Print removing a column from a matrix 
	# y = remove_column(matrix, 0)
	# print("")
	# for i in range (0, dimension):
	# 	print(y[i])

	# Print sum of 2 matrices
	# y = sum(matrix, inv)
	# print("")
	# for i in range (0, dimension):
	# 	print(y[i])

	# Print subtraction of 2 matrices
	# y = subtract(matrix, inv)
	# print("")
	# for i in range (0, dimension):
	# 	print(y[i])

	# Print normalised vector
	# print(matrix[0])
	# v = normalise(matrix[0])
	# print("")
	# for i in range (0, len(v)):
	# 	print(v[i])

	# Print column, row of a vector
	# c = get_column(matrix, 0)
	# print("")
	# print(c)
	# r = get_row(matrix, 1)
	# print("")
	# print(r)

	# Print u of matrix
	# u = find_u(matrix)
	# print("U is:")
	# for i in range (0, len(u)):
	# 	print(u[i])

	# print("U should be:")
	# print("[12, -69, -58/5]\n[6, 158, 6/5]\n[-4, 30,-33]")

	# q = find_q(matrix)
	# print("Q is:")
	# for i in range (len(q)):
	# 	print(q[i])

	# Print QR Decomposition
	# q = []
	# r = []
	# qr_decomp(matrix, q, r)
	# print(q)
	# print(r)
	print("")
	q = find_q(matrix)
	print("Q IS:")

	for i in range (len(q)):
		print(q[i])
	print("")

	r = find_r(matrix, q)
	print("R IS:")
	for i in range (len(r)):
		print(r[i])
	print("")


	# Print power iteration of a matrix
	v = []
	v.append(1)
	for i in range (len(matrix) - 1):
		v.append(0)

	print("v is:")
	print(v)

	eigenvalue = power_it(matrix, v, 0.00001, None)

	print("eigenvalue is:")
	print(eigenvalue)

# ==============================================================================
# call main if executed as script
if __name__ == '__main__':
    sys.exit(main())


