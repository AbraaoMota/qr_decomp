import random
import sys
import os
import argparse
import copy


#======================================================================

class Matrix:

	def __init__(self, m):
		self.m = []

	# Matrix function utilities
	def transpose(self):
		mT = []
		for column in range (0, len(self.m[0])):
			rowT = []
			for row in range (0, len(self.m)):
				rowT.append(self.m[row][column])
			mT.append(rowT)
		self.m = mT
		#return self.m




	def remove_row(self, row):
		#n = copy(m)
		self.m.pop(row)
		#return self.

	def remove_column(self, column):
		#n = m
		for i in range (0, len(self.m)):
			self.m[i].pop(column)
		#return n

	def determinant(m):
		if len(m) == 2:
			return ((m[0][0] * m[1][1]) - (m[0][1] * m[1][0]))
		else:
			sum = 0
			for column in range (0, len(m[0])):
				num = m[0][column]
				newDet = remove_row(m, 0)
				newDet = remove_column(newDet, 0)
				if (column + 1) % 2 == 0:
					sum +=  num * determinant(newDet)
				else:
					sum -= num * determinant(newDet)







	#def inverse(m):



	# def sum(m1, m2):

	# def normalise(v):

	# def dot_product(m1, m2):

def multiply(m1, m2):
	m3 = []
	for rowM in range (0, len(m1.m)):
		row = []
		for columnM in range (0, len(m2.m)):
			product = 0
			for c in range(0, len(m1.m[0])):
				product += m1.m[rowM][c] * m2.m[c][columnM]
			row.append(product)
		m3.append(row)
	return m3


def parity_matrix(dimension):
	matrix = []
	for r in range(0, dimension):
		row = []	
		for c  in range(0, dimension):
			if (c + r + 1) % 2 == 0:
				row.append(-1)
			else:
				row.append(1)
		matrix.append(row)
	return matrix


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
	#matrix = []
	matrix = Matrix([])
	dimension = int(args.n)

	# Parse numbers into the matrix if file given
	if args.fileGiven:
		sourceFile = open(args.filePath, 'r')
		for line in sourceFile:
			numberStrs = line.split()
			nums = [int(x) for x in numberStrs]
			matrix.m.append(nums)		
		sourceFile.close()

	# Create our own random symmetric matrix given a dimension	
	else:
		matrix = create_symmetric_matrix(dimension)

	
	# Print the matrix
	for i in range (0,dimension):
		print(matrix.m[i])

	
	# Print determinant of matrix
	# t = determinant(matrix)
	# print("")
	# print(t)


	# Print transpose of matrix
	# t = transpose(matrix)
	# print("")
	# for i in range (0,dimension):
	# 	print(t[i])

	# Print multiplication of matrices
	# m3 = multiply(matrix, t)
	# print("")
	# for i in range (0,dimension):
	# 	print(m3[i])

	# Print determinant of matrix 

	# Print parity matrix of size dimension
	# d = determinant(matrix)
	# print("")
	# print(d)

	# parity = parity_matrix(dimension)
	# print("")
	# for i in range (0, dimension):
	# 	print(parity[i])

	# Print removing a row from a matrix 
	# matrix.remove_row(1)
	# print("")
	# for i in range (0, dimension - 1):
	# 	print(n[i])

	# print("")

	# for i in range (0,dimension-1):
	# 	print(matrix.m[i])
	# Print removing a column from a matrix 
	# y = remove_column(matrix, 1)
	# print("")
	# for i in range (0, dimension):
	# 	print(y[i])







# ==============================================================================
# call main if executed as script
if __name__ == '__main__':
    sys.exit(main())


