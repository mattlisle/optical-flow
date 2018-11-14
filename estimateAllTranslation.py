'''
  File name: estimateAllTranslation.py
  Author: 
  Date created: 11/4/2018

	(INPUT) startXs: N × F matrix representing the starting X coordinates of all the features in the first frame for all the bounding boxes
	(INPUT) startYs: N × F matrix representing the starting Y coordinates of all the features in the first frame for all the bounding boxes
	(INPUT) img1: H × W × 3 matrix representing the first image frame
	(INPUT) img2: H × W × 3 matrix representing the second image frame
	(OUTPUT) newXs: N × F matrix representing the new X coordinates of all the features in all the bounding boxes
	(OUTPUT) newYs: N × F matrix representing the new Y coordinates of all the features in all the bounding boxes
'''


def estimateAllTranslation(startXs, startYs, img1, img2):
	import numpy as np
	from helpers import rgb2gray
	from scipy import signal

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

	# Initialize our outputs
	newXs = np.zeros(startXs.shape)
	newYs = np.zeros(startYs.shape)

	# Calculate the gradients
	Ix = np.gradient(gray1, axis=1)
	Iy = np.gradient(gray1, axis=0)
	It = gray2 - gray1

	# Pull out parameters for looping
	N, F = startXs.shape
	iterations = 1

	# ---------- Part 2: Caluclate the feature translations ---------- #

	# Use these gradients to find the new locations of the feature points
	# I'm going to not put this into a second function to reduce runtime
	A = np.zeros((window**2, 2))
	b = np.zeros((window**2, 1))

	# For now just running one iteration, not sure where the iterations are supposed to happen
	for k in range(iterations):
		thisx = np.zeros(startXs.shape)
		thisy = np.zeros(startYs.shape)
		for i in range(N):
			for j in range(F):

				# Get our feature location
				fx = int(startXs[i, j]) + pad
				fy = int(startYs[i, j]) + pad

				# Build A and b from A*[u; v] = b centered around the feature location
				A[:, 0] = Ix[fy - pad: fy + pad + 1, fx - pad: fx + pad + 1].reshape(window**2)
				A[:, 1] = Iy[fy - pad: fy + pad + 1, fx - pad: fx + pad + 1].reshape(window**2)
				b[:, 0] = It[fy - pad: fy + pad + 1, fx - pad: fx + pad + 1].reshape(window**2)

				# Solve for [u; v]
				translation = np.matmul(np.matmul(np.linalg.inv(np.matmul(A.T, A)), A.T), -b)

				# Save our result into our output
				thisx[i, j] = startXs[i, j] + translation[0]
				thisy[i, j] = startYs[i, j] + translation[1]
		# # Warp the image and start over again
		# gray1 = warp_image(gray1, thisx, thisy, startXs, startYs)

		# # Update our starting locations
		# startXs = np.copy(thisx)
		# startYs = np.copy(thisy)
	
	newXs = thisx
	newYs = thisy
	return newXs, newYs	
