'''
  File name: calculateError.py
  Author: Matt
  Date created: 11/16/2018

	(INPUT) 
	(OUTPUT) 
'''


def calculateError(startXs, startYs, newXs, newYs, img1, img2, Ix, Iy, box):
	import numpy as np
	from helpers import rgb2gray
	from helpers import interp2
	from scipy import signal
	import matplotlib.pyplot as plt
	from scipy.optimize import least_squares
	from helpers import inlier_cost_func
	from helpers import warp_image

	# This will fail if there is overlap between the boxes

	max_dist = 4
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

	indexer = np.all(np.stack([newXs > xmin + pad, newXs < xmax - pad, newYs > ymin + pad, newYs < ymax - pad], axis=0), axis=0)
	distances = np.sqrt(np.square(newXs - startXs) + np.square(newYs - startYs))
	avg_dist = np.mean(distances)
	std_dist = np.std(distances)
	indexer = np.logical_and(indexer, np.logical_and(distances < avg_dist + 4*std_dist, distances > avg_dist - 2*std_dist))
	# avg_dist = np.mean(distances)
	# std_dist = np.std(distances)
	# indexer = np.logical_and(distances < avg_dist + 1.8*std_dist, distances > avg_dist - 1.8*std_dist)
	if not np.all(indexer):
		removed = len(indexer) - np.sum(indexer)
		total = len(indexer)
		print("Removed %d out of %d points" % (removed, total))

	ux = startXs[indexer]
	uy = startYs[indexer]
	vx = newXs[indexer]
	vy = newYs[indexer]

	# Form our initial and final feature points in homogeneous coordinates
	N = len(ux)
	u = np.stack([ux, uy, np.ones(N)])
	v = np.stack([vx, vy, np.ones(N)])

	T = least_squares(inlier_cost_func, np.identity(3)[:2].reshape(6), args=(u, v))["x"].reshape(2, 3)
	T = np.concatenate((T, np.array([[0, 0, 1]])))

	newXs = np.matmul(T, u)[0]
	newYs = np.matmul(T, u)[1]

	target_area = target[ymin: ymax, xmin: xmax]
	source_area = source[ymin: ymax, xmin: xmax]

	warped_area = warp_image(source, T, xmin, xmax, ymin, ymax)
	source_warped[ymin: ymax, xmin: xmax] = warped_area

	Ix_area = warp_image(Ix, T, xmin, xmax, ymin, ymax)
	Ix[ymin: ymax, xmin: xmax] = Ix_area

	Iy_area = warp_image(Iy, T, xmin, xmax, ymin, ymax)
	Iy[ymin: ymax, xmin: xmax] = Iy_area

	interpx = np.array([newXs])
	interpy = np.array([newYs])
	values_this = interp2(source_warped, interpx, interpy).reshape(len(newXs))
	values_next = interp2(target, interpx, interpy).reshape(len(newXs))

	error = np.sum(np.square(values_next - values_this)) / len(newXs)

	return error, source_warped, indexer, Ix, Iy, newXs, newYs
