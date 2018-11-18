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

def applyGeometricTransformation(startXs, startYs, newXs, newYs, bbox, img, target, thresh):
	import numpy as np
	from helpers import inlier_cost_func
	from scipy.optimize import least_squares
	from helpers import warp_image
	from helpers import rgb2gray
	import matplotlib.pyplot as plt
	from helpers import inlier_cost_func_translation
	from ransac_est_affine import ransac_est_affine

	F = len(bbox)
	max_dist = 100
	pad = 5
	gray = rgb2gray(img)
	output = np.copy(gray)
	target = rgb2gray(target)
	h, w = gray.shape

	for i in range(F):

		# If the item went of the screen, its bounding box could be empty
		if not np.any(bbox[i]):
			continue

		# --------- Part 1: Estimate the homography for a given bounding box ---------- #

		# Remove points that lie outside the current box
		# medium difficulty
		# xmin = max([np.amin(bbox[i, :, 0]), np.amin(newXs[i]) - 2.5*pad])
		# xmax = min([np.amax(bbox[i, :, 0]), np.amax(newXs[i]) + 2.5*pad])
		# # print(np.amax(bbox[i, :, 0]), np.amax(newXs[i]) + pad)
		# ymin = max([np.amin(bbox[i, :, 1]), np.amin(newYs[i]) - 2.5*pad])
		# ymax = min([np.amax(bbox[i, :, 1]), np.amax(newYs[i]) + 2.5*pad])
		# bbox[i] = np.array([xmin, ymin, xmin, ymax, xmax, ymax, xmax, ymin]).reshape(4, 2)

		# easy difficulty
		xmin = max([np.amin(bbox[i, :, 0]), np.amin(newXs[i]) - 2*pad])
		xmax = min([np.amax(bbox[i, :, 0]), np.amax(newXs[i]) + 2*pad])
		# print(np.amax(bbox[i, :, 0]), np.amax(newXs[i]) + pad)
		ymin = max([np.amin(bbox[i, :, 1]), np.amin(newYs[i]) - 2*pad])
		ymax = min([np.amax(bbox[i, :, 1]), np.amax(newYs[i]) + 2*pad])
		bbox[i] = np.array([xmin, ymin, xmin, ymax, xmax, ymax, xmax, ymin]).reshape(4, 2)

		# xmin = np.amin(bbox[i, :, 0])
		# xmax = np.amax(bbox[i, :, 0])
		# ymin = np.amin(bbox[i, :, 1])
		# ymax = np.amax(bbox[i, :, 1])

		# indexer = np.all(np.stack([newXs[i] > xmin, newXs[i] < xmax, newYs[i] > ymin, newYs[i] < ymax], axis=0))
		indexer = np.ones(len(startXs[i]), dtype=bool)

		ux = startXs[i][indexer]
		uy = startYs[i][indexer]
		vx =   newXs[i][indexer]
		vy =   newYs[i][indexer]

		# # Now remove points who traveled to far between frames
		# distances = np.sqrt(np.square(vx - ux) + np.square(vy - uy))
		# indexer = distances < max_dist 
		# indexer = np.ones(len(startXs[i]), dtype=bool)
		# ux = ux[indexer]
		# uy = uy[indexer]
		# vx = vx[indexer]
		# vy = vy[indexer]

		# Form our initial and final feature points in homogeneous coordinates
		N = len(ux)
		u = np.stack([ux, uy, np.ones(N)])
		v = np.stack([vx, vy, np.ones(N)])

		# Use RANSAC to estimate T
		# H, inliers = ransac_est_affine(ux, uy, vx, vy, thresh)
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
			output = gray
			newXs[i] = vx
			newYs[i] = vy
			continue

		# Restrict the bounding box to the image frame
		corners[corners < 0] = 0
		corners[0][corners[0] >= w] = w - 1
		corners[1][corners[1] >= h] = h - 1

		# Update the corners of the box
		bbox[i, ...] = corners[:2].T

		# --------- Part 3: Warp the bounding box areas of the image ---------- #

		# Create a sub image that matches the area of the new bounding box and warp that
		# subimg = target[ymin - pad: ymax + pad + 1, xmin - pad: xmax + pad + 1]  # For debugging
		# initial = gray[ymin - pad: ymax + pad + 1, xmin - pad: xmax + pad + 1]
		# warped = warp_image(gray, H, xmin - pad, xmax + pad + 1, ymin - pad, ymax + pad + 1)

		# For now, the warped image assumes that its inside the boundary of the image
		# output[ymin - pad: ymax + pad + 1, xmin - pad: xmax + pad + 1] = warped
		output = gray

		# Update the feature entries to remove outliers
		newXs[i] = vx
		newYs[i] = vy

	return newXs, newYs, bbox, output