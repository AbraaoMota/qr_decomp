import random
import sys
import os
import argparse

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

#def inverse(m):

def multiply(m1, m2):
	m3 = []
	for row in range (0, len(m1[0])):
		row3 = []
		for column in range (0, len(m1)):
			row3.append(m1[row][column] * m2[row][column])
		m3.append(row3)
	return m3


# def sum(m1, m2):

#def determinant(m):
	#if len(m) == 2:
		#return ((m[0][0] * m[1][1]) - (m[0][1] * m[1][0]))
	#else:


# def normalise(v):

# def dot_product(m1, m2):




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
	matrix = []
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
	t = transpose(matrix)
	print("")
	for i in range (0,dimension):
		print(t[i])

	# Print multiplication of matrices
	m3 = multiply(matrix, t)
	print("")
	for i in range (0,dimension):
		print(m3[i])		

# ==============================================================================
# call main if executed as script
if __name__ == '__main__':
    sys.exit(main())


