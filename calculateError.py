'''
  File name: calculateError.py
  Author: Nikhil, Shiv, Matt
  Date created: 11/16/2018

	(INPUT) startXs: object array representing the starting X coordinates of all the features in the first frame for all the bounding boxes
	(INPUT) startYs: object array representing the starting Y coordinates of all the features in the first frame for all the bounding boxes
	(INPUT) newXs: object array matrix representing the second X coordinates of all the features in the first frame for all the bounding boxes
	(INPUT) newYs: object array matrix representing the second Y coordinates of all the features in the first frame for all the bounding boxes
	(INPUT) img1: H × W × 3 matrix representing the first image frame
	(INPUT) img2: H × W × 3 matrix representing the next image frame
	(INPUT) Ix: H × W matrix representing the gradient along the X-direction
	(INPUT) Iy: H × W matrix representing the gradient along the Y-direction
	(INPUT) box: 4 × 2 matrix representing the four corners of the bounding box
	(INPUT) params: constants specified depending on the rawVideo input
	(OUTPUT) error: squared distance between tracked features in source and target images
	(OUTPUT) source_warped: H x W matrix representing the source image after the transformation is applied to each bounding box
	(OUTPUT) indexer: boolean array where a value of True represents an inler 
	(OUTPUT) Ix: H × W matrix representing the gradient along the X-direction
	(OUTPUT) Iy: H × W matrix representing the gradient along the Y-direction
	(OUTPUT) newXs: object array matrix representing the second X coordinates of all the features in the first frame for all the bounding boxes
	(OUTPUT) newYs: object array matrix representing the second Y coordinates of all the features in the first frame for all the bounding boxes
'''


def calculateError(startXs, startYs, newXs, newYs, img1, img2, Ix, Iy, box, params):
	import numpy as np
	from helpers import rgb2gray
	from helpers import interp2
	from scipy import signal
	from scipy.optimize import least_squares
	from helpers import inlier_cost_func
	from helpers import warp_image

	# Extract parameters
	max_dist = params[0]
	k1 = params[1]
	k2 = params[2]
	k3 = params[3]
	k4 = params[4]
	pad = 5

	source = rgb2gray(img1)
	target = rgb2gray(img2)
	source_warped = np.copy(source)

	h, w = source.shape

	# Get the boundaries of bounding box
	xmin = max([np.amin(box[:, 0]) - pad, 0])
	xmax = min([np.amax(box[:, 0]) + pad + 1, w])
	ymin = max([np.amin(box[:, 1]) - pad, 0])
	ymax = min([np.amax(box[:, 1]) + pad + 1, h])

	# Outlier handling
	indexer = np.all(np.stack([newXs > xmin + pad, newXs < xmax - pad, newYs > ymin + pad, newYs < ymax - pad], axis=0), axis=0)
	distances = np.sqrt(np.square(newXs - startXs) + np.square(newYs - startYs))
	avg_dist = np.mean(distances)
	std_dist = np.std(distances)
	if avg_dist != 0:
		indexer = np.logical_and(indexer, np.logical_and(distances < min([k1*avg_dist + k2*std_dist, max_dist]), distances > k3*avg_dist - k4*std_dist))

	# Generate vectors of inliers for calculating the transformation
	ux = startXs[indexer]
	uy = startYs[indexer]
	vx = newXs[indexer]
	vy = newYs[indexer]

	# Form our initial and final feature points in homogeneous coordinates
	N = len(ux)
	u = np.stack([ux, uy, np.ones(N)])
	v = np.stack([vx, vy, np.ones(N)])

	# Calculate the transformation via least squares
	T = least_squares(inlier_cost_func, np.identity(3)[:2].reshape(6), args=(u, v))["x"].reshape(2, 3)
	T = np.concatenate((T, np.array([[0, 0, 1]])))
	newXs = np.matmul(T, u)[0]
	newYs = np.matmul(T, u)[1]

	# Warp img1, Ix and Iy based on calculated transformation
	target_area = target[ymin: ymax, xmin: xmax]
	source_area = source[ymin: ymax, xmin: xmax]

	warped_area = warp_image(source, T, xmin, xmax, ymin, ymax)
	source_warped[ymin: ymax, xmin: xmax] = warped_area

	Ix_area = warp_image(Ix, T, xmin, xmax, ymin, ymax)
	Ix[ymin: ymax, xmin: xmax] = Ix_area

	Iy_area = warp_image(Iy, T, xmin, xmax, ymin, ymax)
	Iy[ymin: ymax, xmin: xmax] = Iy_area

	# Calculate the error per feature point
	interpx = np.array([newXs])
	interpy = np.array([newYs])
	values_this = interp2(source_warped, interpx, interpy).reshape(len(newXs))
	values_next = interp2(target, interpx, interpy).reshape(len(newXs))
	error = np.sum(np.square(values_next - values_this)) / len(newXs)

	return error, source_warped, indexer, Ix, Iy, newXs, newYs
