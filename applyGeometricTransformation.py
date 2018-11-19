'''
	File name: applyGeometricTransformation.py
	Author: Nikhil, Shiv, Matt
	Date created: 11/4/2018

	(INPUT) startXs: object array representing the starting X coordinates of all the features in the first frame for all the bounding boxes
	(INPUT) startYs: object array representing the starting Y coordinates of all the features in the first frame for all the bounding boxes
	(INPUT) newXs: object array matrix representing the second X coordinates of all the features in the first frame for all the bounding boxes
	(INPUT) newYs: object array matrix representing the second Y coordinates of all the features in the first frame for all the bounding boxes
	(INPUT) bbox: F × 4 × 2 matrix representing the four new corners of the bounding box where F is the number of detected objects
	(INPUT) img: H × W × 3 matrix representing the first image frame
	(INPUT) k_pad: constant specified depending on the rawVideo input
	(OUTPUT) Xs: object array matrix representing the X coordinates of the remaining features in all the bounding boxes after eliminating outliers
	(OUTPUT) Ys: object array matrix representing the Y coordinates of the remaining features in all the bounding boxes after eliminating outliers
	(OUTPUT) newbbox: F × 4 × 2 the bounding box position after geometric transformation
'''

def applyGeometricTransformation(startXs, startYs, newXs, newYs, bbox, img, k_pad):
	import numpy as np
	from helpers import inlier_cost_func
	from helpers import rgb2gray
	from scipy.optimize import least_squares

	F = len(bbox)
	pad = 5
	gray = rgb2gray(img)
	output = np.copy(gray)
	h, w = gray.shape

	for i in range(F):

		# If the item went of the screen, its bounding box could be empty
		if not np.any(bbox[i]):
			continue

		# --------- Part 1: Estimate the homography for a given bounding box ---------- #

		# Squeeze the box to the features within a margin determined by k*pad
		xmin = max([np.amin(bbox[i, :, 0]), np.amin(newXs[i]) - k_pad*pad])
		xmax = min([np.amax(bbox[i, :, 0]), np.amax(newXs[i]) + k_pad*pad])
		ymin = max([np.amin(bbox[i, :, 1]), np.amin(newYs[i]) - k_pad*pad])
		ymax = min([np.amax(bbox[i, :, 1]), np.amax(newYs[i]) + k_pad*pad])
		bbox[i] = np.array([xmin, ymin, xmin, ymax, xmax, ymax, xmax, ymin]).reshape(4, 2)

		ux = np.copy(startXs[i])
		uy = np.copy(startYs[i])
		vx =   np.copy(newXs[i])
		vy =   np.copy(newYs[i])

		# Form our initial and final feature points in homogeneous coordinates
		N = len(ux)
		u = np.stack([ux, uy, np.ones(N)])
		v = np.stack([vx, vy, np.ones(N)])

		# Calculate the transformation
		H = least_squares(inlier_cost_func, np.identity(3)[:2].reshape(6), args=(u, v))["x"].reshape(2, 3)
		H = np.concatenate((H, np.array([[0, 0, 1]])))

		# --------- Part 2: Update the ith bounding box ---------- #
		
		# Apply the homography to the corners
		corners = np.stack([bbox[i].T[0], bbox[i].T[1], np.ones(4)])
		corners = np.matmul(H, corners)
		corners = corners / corners[2]  # unnecessary for affine transformations
		
		# If the object has passed out of the image frame, get rid of it
		if np.any(np.logical_or(corners[0] >= w, corners[0] < 0)) and np.any(np.logical_or(corners[1] >= h, corners[1] < 0)):
			bbox[i] = np.zeros((4, 2))
			newXs[i] = vx
			newYs[i] = vy
			continue

		# Restrict the bounding box to the image frame
		corners[corners < 0] = 0
		corners[0][corners[0] >= w] = w - 1
		corners[1][corners[1] >= h] = h - 1

		# Update the corners of the box
		bbox[i, ...] = corners[:2].T

		# Update the feature entries to remove outliers
		newXs[i] = vx
		newYs[i] = vy

	return newXs, newYs, bbox