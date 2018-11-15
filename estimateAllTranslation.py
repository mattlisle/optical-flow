'''
  File name: estimateAllTranslation.py
  Author: 
  Date created: 11/4/2018

	(INPUT) startXs: object array representing the starting X coordinates of all the features in the first frame for all the bounding boxes
	(INPUT) startYs: object array representing the starting Y coordinates of all the features in the first frame for all the bounding boxes
	(INPUT) img1: H × W × 3 matrix representing the first image frame
	(INPUT) img2: H × W × 3 matrix representing the second image frame
	(OUTPUT) newXs: object array representing the new X coordinates of all the features in all the bounding boxes
	(OUTPUT) newYs: object array representing the new Y coordinates of all the features in all the bounding boxes
'''


def estimateAllTranslation(startXs, startYs, img1, img2):
	import numpy as np
	from helpers import rgb2gray
	from helpers import interp2
	from scipy import signal
	import matplotlib.pyplot as plt

	# ---------- Part 1: Setup ---------- #

	# Get images computed to grayscale
	# For now I'm going to pad the images symmetrically to get W for edges and corners
	# It will just be important to remember that padding's there when solving for pixel locations
	window = 11
	pad = int((window - 1) / 2)
	# gray1 = np.pad(rgb2gray(img1), ((pad, pad), (pad, pad)), mode="symmetric")
	# gray2 = np.pad(rgb2gray(img2), ((pad, pad), (pad, pad)), mode="symmetric")

	# Blur the images to get better optical flow results
	Gx = signal.gaussian(window, 1.4).reshape(1, window)
	Gy = signal.gaussian(window, 1.4).reshape(window, 1)
	gray1 = signal.convolve2d(rgb2gray(img1), Gx, mode="full", boundary="symm")
	gray1 = signal.convolve2d(		   gray1, Gy, mode="full", boundary="symm")
	gray2 = signal.convolve2d(rgb2gray(img2), Gx, mode="full", boundary="symm")
	gray2 = signal.convolve2d(		   gray2, Gy, mode="full", boundary="symm")
	# gray1 = img1
	# gray2 = img2

	# Calculate the gradients
	# kx = np.array([[1, -1]])
	# ky = np.array([[1],[-1]])
	Ix = np.gradient(gray1, axis=1)
	Iy = np.gradient(gray1, axis=0)
	It = gray2 - gray1  # np.gradient(np.array([gray1, gray2]), axis=0)
	# plt.imshow(gray2 - gray1, cmap="gray")
	# plt.show()
	# print("Total error for this iteration", np.sum(np.abs(It)))
	# plt.imshow(It, cmap="gray")
	# plt.show()

	# Pull out parameters for looping
	F = len(startXs)

	# Initialize our outputs
	newXs = np.zeros(F, dtype=object)
	newYs = np.zeros(F, dtype=object)

	# ---------- Part 2: Caluclate the feature translations ---------- #

	# Use these gradients to find the new locations of the feature points
	# I'm going to not put this into a second function to reduce runtime
	A = np.zeros((window**2, 2))
	b = np.zeros((window**2, 1))

	# For now just running one iteration, not sure where the iterations are supposed to happen
	for i in range(F):
		N = len(startXs[0])
		newXs[i] = np.zeros(N)
		newYs[i] = np.zeros(N)
		for j in range(N):

			# Get our feature location
			fx = startXs[i][j] + pad
			fy = startYs[i][j] + pad

			# Generate a meshgrid for interpolating
			meshx, meshy = np.meshgrid(np.arange(window), np.arange(window))
			meshx = meshx + fx
			meshy = meshy + fy

			# Build A and b from A*[u; v] = b centered around the feature location
			A[:, 0] = interp2(Ix, meshx, meshy).reshape(window**2)  # Ix[fy - pad: fy + pad + 1, fx - pad: fx + pad + 1].reshape(window**2)
			A[:, 1] = interp2(Iy, meshx, meshy).reshape(window**2)  # Iy[fy - pad: fy + pad + 1, fx - pad: fx + pad + 1].reshape(window**2)
			b[:, 0] = interp2(It, meshx, meshy).reshape(window**2)  # It[fy - pad: fy + pad + 1, fx - pad: fx + pad + 1].reshape(window**2)

			# error = np.sum(np.abs(b))
			# if j == 0:
			# 	print("Error for box %d and feature %d: " % (i, j), error)

			# Solve for [u; v]
			translation = np.matmul(np.matmul(np.linalg.inv(np.matmul(A.T, A)), A.T), -b)
			# print(translation)

			# Save our result into our output
			newXs[i][j] = startXs[i][j] + translation[0]
			newYs[i][j] = startYs[i][j] + translation[1]

	return newXs, newYs	
