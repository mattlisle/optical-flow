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

	# Get images computed to grayscale
	# For now I'm going to pad the images symmetrically to get W for edges and corners
	# It will just be important to remember that padding's there when solving for pixel locations
	pad = 10
	gray1 = np.pad(rgb2gray(img1), ((pad, pad), (pad, pad)), mode="symmetric")
	gray2 = np.pad(rgb2gray(img2), ((pad, pad), (pad, pad)), mode="symmetric")
	imgs = np.array([gray1, gray2])

	# Calculate the gradients
	Ix = np.gradient(gray1, axis=1)
	Iy = np.gradient(gray1, axis=0)
	It = np.gradient(imgs, axis=0)

