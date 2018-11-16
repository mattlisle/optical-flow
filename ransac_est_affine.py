'''
	File name: ransac_est_affine.py
	Author: Matt
	Date created: 11/15/2018

	(INPUT) x1, y1, x2, y2: N x 1 arrays representing corresponding feature points
	(INPUT) thresh: float indicating tolerance of (x2 - T*x1)^2
	(OUTPUT) T: affine transformation between these points
'''

def ransac_est_affine(x1, y1, x2, y2, thresh):
	import numpy as np
	from scipy.optimize import least_squares
	from helpers import inlier_cost_func
	from helpers import est_affine

	# Number of RANSAC trials
	t = 20

	# Length of matching points arrays
	n = len(x1)
	z = np.ones(n)

	inlier_ind = np.array([])

	for i in range(t):
		# Choose 3 points randomly and estimate a homography
		choices = np.random.choice(n, 3, replace=False)
		T = est_affine(np.stack([x1[choices], y1[choices], np.ones(3)]), np.stack([x2[choices], y2[choices], np.ones(3)]))

		# Use the homography to transform points from img1 to estimated postions in img2
		estimates = np.matmul(T, np.stack([x1, y1, z]))

		# Normalize the estimates and extract x and y
		estimates = estimates / estimates[-1]
		x_est = estimates[0]
		y_est = estimates[1]

		# Compute sum of squared distances (That's what the assignment said, but I don't see why we would sum them?)
		distances = np.sqrt(np.square(x2 - x_est) + np.square(y2 - y_est))
		ind = np.where(distances < thresh)[0]

    	# If we did better this time, save the indices
		if len(ind) > len(inlier_ind):
			found = len(ind)
			print("Found %d matches..." % found, end="\r", flush=True)
			inlier_ind = ind
			dist = distances
			Tbest = T

	# Run least squares on the inliers to get the most reliable estimate of H
	xin = np.stack([x1[inlier_ind], y1[inlier_ind], np.ones(len(inlier_ind))])
	yin = np.stack([x2[inlier_ind], y2[inlier_ind], np.ones(len(inlier_ind))])
	T = least_squares(inlier_cost_func, Tbest[:2].reshape(6), args=(xin, yin))["x"].reshape(2, 3)

	print("Found %d matches..." % found)
	return np.concatenate((T, np.array([[0, 0, 1]]))), inlier_ind
