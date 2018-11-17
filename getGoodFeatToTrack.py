'''
  File name: getGoodFeatToTrack.py
  Author: Shiv
  Date created: 11/16/2018
  Similar to getFeatures but uses opencv's goodFeaturesToTrack function
	(INPUT) img: H × W x 3 matrix representing the BGR input image
	(INPUT) bbox: F × 4 × 2 matrix representing the four corners of the bounding box where F is the number of
	objects you would like to track
	(OUTPUT) x: N × F matrix representing the N row coordinates of the features across F objects
	(OUTPUT) y: N × F matrix representing the N column coordinates of the features across F objects

'''

def getGoodFeatToTrack(img, bbox, num, quality, minDist):
	import numpy as np
	from skimage.feature import corner_shi_tomasi
	from helpers import anms
	import matplotlib.pyplot as plt
	import cv2

	# Initialize our outputs
	x = np.zeros(bbox.shape[0], dtype=object)
	y = np.zeros(bbox.shape[0], dtype=object)

	for i in range(bbox.shape[0]):
		# Save our offsets from the bbox array, not necessary but improves readability
		xmin = np.amin(bbox[i, :, 0])
		xmax = np.amax(bbox[i, :, 0])
		ymin = np.amin(bbox[i, :, 1])
		ymax = np.amax(bbox[i, :, 1])

		

		# For debugging: Show the what's inside the bounding box
		# plt.imshow(subimg[p:-p, p:-p])
		# plt.show()
		# have to use openCV to convert to gray or else types dont match
		gray = cv2.cvtColor(img[xmin:xmax,ymin:ymax,:], cv2.COLOR_BGR2GRAY)
		# get the shi-tomassi corners 
		# Params: img, number of corners, corner quality, miniumum distance from each other
		pts = cv2.goodFeaturesToTrack(gray, num,quality,minDist)


		x[i] = pts[:,:,0]+xmin
		y[i] = pts[:,:,1]+ymin
	return x, y