import random
import argparse

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







################################

if __name__ == "__main__":
	matrix = create_symmetric_matrix(2)
	print(matrix)	

	parser = argparse.ArgumentParser(description='Choose file or dimension of matrix')
	parser.add_argument('integers', metavar='N', type=int, nargs='+',
                   help='an integer for the accumulator')
	parser.add_argument('--sum', dest='accumulate', action='store_const',
                   const=sum, default=max,
                   help='sum the integers (default: find the max)')

	args = parser.parse_args()
	print args.accumulate(args.integers)
