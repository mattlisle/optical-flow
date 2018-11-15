'''
	File name: est_affine.py
	Author: Matt
	Date created: 11/15/2018

	(INPUT) x1: 3 x 3 array of 3 feature points in original image
	(INPUT) x2: 3 x 3 array of 3 feature points after feature translation
	(OUTPUT) T: affine transformation between these points
'''

def est_affine(x1, x2):
	import numpy as np

	A = np.zeros((6, 6))

	A[:3] = np.pad(x1.T, ((0, 0), (0, 3)), mode="constant")
	A[3:] = np.pad(x1.T, ((0, 0), (3, 0)), mode="constant")

	b = np.concatenate((x2[0], x2[1])).reshape(6, 1)

	try:
		T = np.concatenate((np.matmul(np.linalg.inv(A), b).reshape(2, 3), np.array([[0, 0, 1]])))
	except LinAlgError:
		T = np.identity(3)

	return T