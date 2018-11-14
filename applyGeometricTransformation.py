'''
	File name: applyGeometricTransformation.py
	Author: 
	Date created: 11/4/2018

	(INPUT) startXs: object array representing the starting X coordinates of all the features in the first frame for all the bounding boxes
	(INPUT) startYs: object array representing the starting Y coordinates of all the features in the first frame for all the bounding boxes
	(INPUT) newXs: object array matrix representing the second X coordinates of all the features in the first frame for all the bounding boxes
	(INPUT) newYs: object array matrix representing the second Y coordinates of all the features in the first frame for all the bounding boxes
	(INPUT) bbox: F × 4 × 2 matrix representing the four new corners of the bounding box where F is the number of detected objects
	(OUTPUT) Xs: object array matrix representing the X coordinates of the remaining features in all the bounding boxes after eliminating outliers
	(OUTPUT) Ys: object array matrix representing the Y coordinates of the remaining features in all the bounding boxes after eliminating outliers
	(OUTPUT) newbbox: F × 4 × 2 the bounding box position after geometric transformation
	(OUTPUT) gray: the grayscale warped image for the next iteration of translation estimation
'''

def applyGeometricTransformation(startXs, startYs, newXs, newYs, bbox, img):
	import numpy as np
	from helpers import inlier_cost_func
	from scipy.optimize import least_squares
	from helpers import warp_image
	from helpers import rgb2gray
	import matplotlib.pyplot as plt

	F = len(startXs)
	max_dist = 4
	pad = 5
	gray = rgb2gray(img)

	for i in range(F):
		N = len(startXs[0])

		# --------- Part 1: Estimate the homography for a given bounding box ---------- #

		# Remove points that lie outside the current box
		xmin = np.amin(bbox[i, :, 0])
		xmax = np.amax(bbox[i, :, 0])
		ymin = np.amin(bbox[i, :, 1])
		ymax = np.amax(bbox[i, :, 1])

		indexer = np.all(np.stack([newXs[i] > xmin, newXs[i] < xmax, newYs[i] > ymin, newYs[i] < ymax], axis=0))

		ux = startXs[i][indexer]
		uy = startYs[i][indexer]
		vx =   newXs[i][indexer]
		vy =   newYs[i][indexer]

		# Now remove points who traveled to far between frames
		distances = np.sqrt(np.square(vx - ux) + np.square(vy - uy))
		indexer = distances < max_dist 
		ux = ux[indexer]
		uy = uy[indexer]
		vx = vx[indexer]
		vy = vy[indexer]

		# Form our initial and final feature points in homogeneous coordinates
		u = np.stack([ux, uy, np.ones(N)])
		v = np.stack([vx, vy, np.ones(N)])

		# Initial guess for the homography will be the identity matrix
		H_init = np.identity(3)[:2]

		# Use least squares to estimate H
		H = least_squares(inlier_cost_func, H_init.reshape(6), args=(v, u))["x"]
		H = np.stack([H[:3], H[3: 6], np.array([0, 0, 1])])
		print(H)

		# --------- Part 2: Update the ith bounding box ---------- #
		
		# Apply the homography to the corners
		corners = np.stack([bbox[i].T[0], bbox[i].T[1], np.ones(4)])
		corners = np.matmul(H, corners)
		corners = corners / corners[2]

		# Force the bbox to be a rectangle
		xmin = int(np.floor(np.amin(corners[0])))
		xmax = int(np.ceil( np.amax(corners[0])))
		ymin = int(np.floor(np.amin(corners[1])))
		ymax = int(np.ceil( np.amax(corners[1])))
		bbox[i, ...] = np.array([xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]).reshape(4,2)

		# --------- Part 3: Warp the bounding box areas of the image ---------- #

		# Create a sub image that matches the area of the new bounding box and warp that
		# subimg = gray[ymin - pad: ymax + pad + 1, xmin - pad: xmax + pad + 1]  # For debugging
		warped = warp_image(gray, H, xmin - pad, xmax + pad + 1, ymin - pad, ymax + pad + 1)

		# For now, the warped image assumes that its inside the boundary of the image
		gray[ymin - pad: ymax + pad + 1, xmin - pad: xmax + pad + 1] = warped

	return newXs, newYs, bbox, gray