import random

# Let a matrix of m rows and n columns be represented by m lists of n items


def create_symmetric_matrix(rows, columns):
	matrix = []
	columns_covered = 0

	for r in range (0, rows):
		row = []
		for c in range (0, columns):
			if columns_covered > 0 and c < columns_covered:
				row.append(matrix[c][r])
			else:
				row.append(random.random())
		matrix.append(row)
		columns_covered += 1

	return matrix

matrix = create_symmetric_matrix(2,2)
print(matrix)
