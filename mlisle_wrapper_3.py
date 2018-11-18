'''
  File name: mlisle_wrapper.py
  Author: Matt
  Date created: 11/4/2018
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from helpers import rgb2gray
from helpers import generate_output_frame
from getFeatures import getFeatures
from calculateError import calculateError
from estimateAllTranslation import estimateAllTranslation
from applyGeometricTransformation import applyGeometricTransformation

imgs = np.array([])
cap = cv2.VideoCapture("Easy.mp4")
ret, img1 = cap.read()
img1 = img1[...,::-1]
h, w, d = img1.shape

# For now, manually draw the bounding box and forget about cv2.boundingRect()
box1 = np.array([287, 187, 397, 187, 397, 264, 287, 264]).reshape(4, 2)
box2 = np.array([227, 127, 279, 127, 279, 172, 227, 172]).reshape(4, 2)
bbox = np.array([box1, box2])
# bbox = np.array([box1])
orig_box = np.copy(bbox)
centers = np.zeros((len(bbox), 2))

orig_box = np.copy(bbox)
centers = np.zeros((len(bbox), 2))
trajectory_indexer = np.zeros((h, w), dtype=bool)

# Get the features from inside the bounding box
x, y = getFeatures(rgb2gray(img1), bbox)

newXs = np.copy(x)
newYs = np.copy(y)

f = 0
frame = generate_output_frame(np.copy(img1), bbox, np.copy(trajectory_indexer), np.copy(newXs), np.copy(newYs))
frame = Image.fromarray(frame)
frame.save("easy_frame%d.jpg" % f)

a = 0
while True:
	f += 1
	a += 1
	if bbox.size:
		if bbox.size < 2:
			print(bbox.size)
		if not f % 8:
			print("Frame: ", f)
			a = 1
			for i in range(len(bbox)):
				orig_box = np.copy(bbox)
			x, y = getFeatures(rgb2gray(img1), bbox)
			newXs = np.copy(x)
			newYs = np.copy(y)

		thresh = .1 + .02 * a

		ret, img2 = cap.read()
		if not ret:
			break
		img2 = img2[...,::-1]

		# Get the new feature locations in the next frame
		updatex, updatey, x, y = estimateAllTranslation(np.copy(newXs), np.copy(newYs), np.copy(x), np.copy(y), np.copy(img1), np.copy(img2), np.copy(bbox))

		for k in range(len(bbox)):
			centers[k] = np.array([np.mean(bbox[k, :, 0]), np.mean(bbox[k, :, 1])]).astype(int)

		# Warp the image for the next iteration
		newXs, newYs, bbox, warped = applyGeometricTransformation(np.copy(x), np.copy(y), updatex, updatey, np.copy(orig_box), np.copy(img1), np.copy(img2), thresh)

		indexer = np.ones(len(bbox), dtype=bool)
		for k in range(len(bbox)):
			if not np.any(bbox[k]) or len(newXs[k]) < 2:
				indexer[k] = False

		bbox = bbox[indexer]
		orig_box = orig_box[indexer]
		newXs = newXs[indexer]
		newYs = newYs[indexer]
		x = x[indexer]
		y = y[indexer]
		centers = centers[indexer]

		for k in range(len(bbox)):
			xcen = int(np.mean(bbox[k, :, 0]))
			ycen = int(np.mean(bbox[k, :, 1]))
			if xcen < w - 2 and xcen > 2 and ycen < h - 2 and ycen > 2:
				num = int(max([abs(xcen - centers[k, 0]), abs(ycen - centers[k, 1])]))
				centerx = np.linspace(centers[k, 0], xcen + 1, num).astype(int)
				centery = np.linspace(centers[k, 1], ycen + 1, num).astype(int)
				if centerx.size > 0 and centery.size > 0:
					trajectory_indexer[centery, centerx] = True
					trajectory_indexer[centery + 1, centerx] = True
					trajectory_indexer[centery, centerx + 1] = True
					trajectory_indexer[centery + 1, centerx + 1] = True
				else:
					trajectory_indexer[ycen, xcen] = True
					trajectory_indexer[ycen + 1, xcen] = True
					trajectory_indexer[ycen, xcen + 1] = True
					trajectory_indexer[ycen + 1, xcen + 1] = True

		frame = generate_output_frame(np.copy(img2), bbox, np.copy(trajectory_indexer), np.copy(newXs), np.copy(newYs))
		frame = Image.fromarray(frame)
		frame.save("easy_frame%d.jpg" % f)
	else:
		ret, img2 = cap.read()
		if not ret:
			break
		img2 = img2[...,::-1]
		frame = Image.fromarray(img2)
		frame.save("easy_frame%d.jpg" % f)

	img1 = np.copy(img2)

cap.release()