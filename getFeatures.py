'''
  File name: getFeatures.py
  Author: 
  Date created: 11/4/2018

	(INPUT) img: H × W matrix representing the grayscale input image
	(INPUT) bbox: F × 4 × 2 matrix representing the four corners of the bounding box where F is the number of
	objects you would like to track
	(OUTPUT) x: N × F matrix representing the N row coordinates of the features across F objects
	(OUTPUT) y: N × F matrix representing the N column coordinates of the features across F objects

'''

def getFeatures(img, bbox):
	import numpy as np
	from skimage.feature import corner_harris
	from helpers import anms
	import matplotlib.pyplot as plt

	# Feature points gotten from image bounding box
	max_pts = 20

	# Initialize our outputs
	x = np.zeros(bbox.shape[0], dtype=object)
	y = np.zeros(bbox.shape[0], dtype=object)

	for i in range(bbox.shape[0]):
		# Save our offsets from the bbox array, not necessary but improves readability
		xmin = np.amin(bbox[i, :, 0])
		xmax = np.amax(bbox[i, :, 0])
		ymin = np.amin(bbox[i, :, 1])
		ymax = np.amax(bbox[i, :, 1])

		# Get the corner strength array from the bouding box area with padding
		p = 10
		subimg = img[ymin - p: ymax + p, xmin - p: xmax + p]
		# print(subimg.shape)

		# For debugging: Show the what's inside the bounding box
		# plt.imshow(subimg[p:-p, p:-p])
		# plt.show()
		
		# Get corner strength matrix
		cimg = corner_harris(subimg, k=0.05, sigma=1)[p: -p, p: -p]

		# Suppress non-maxima	
		x[i], y[i] = anms(cimg, max_pts, xmin, ymin)

	return x, y