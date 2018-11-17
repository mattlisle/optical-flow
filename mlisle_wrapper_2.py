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
from estimateAllTranslation import estimateAllTranslation
from applyGeometricTransformation import applyGeometricTransformation

imgs = np.array([])
cap = cv2.VideoCapture("Medium.mp4")
ret, img1 = cap.read()
img1 = img1[...,::-1]
h, w, d = img1.shape

# For now, manually draw the bounding box and forget about cv2.boundingRect()
box1 = np.array([456, 182, 456, 279, 523, 279, 523, 182]).reshape(4, 2)
bbox = np.array([box1])

orig_box = np.copy(bbox)
centers = np.zeros((len(bbox), 2))
trajectory_indexer = np.zeros((h, w), dtype=bool)

f = 0
frame = generate_output_frame(np.copy(img1), bbox, np.copy(trajectory_indexer))
frame = Image.fromarray(frame)
frame.save("easy_frame%d.jpg" % f)

# Get the features from inside the bounding box
x, y = getFeatures(rgb2gray(img1), bbox)

newXs = np.copy(x)
newYs = np.copy(y)

a = 0
while ret:
	f += 1
	a += 1
	if not f % 5:
		a = 1
		for i in range(len(bbox)):
			xmin = np.sort(bbox[i, :, 0])[0]
			xmax = np.sort(bbox[i, :, 0])[3]
			ymin = np.sort(bbox[i, :, 1])[0]
			ymax = np.sort(bbox[i, :, 1])[3]
			bbox[i, ...] = np.array([xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]).reshape(4,2)
			orig_box = np.copy(bbox)
		x, y = getFeatures(rgb2gray(img1), bbox)
		newXs = np.copy(x)
		newYs = np.copy(y)

	thresh = .2 + .02 * a

	ret, img2 = cap.read()
	img2 = img2[...,::-1]

	iterations = 1

	# Get the new feature locations in the next frame
	updatex, updatey = estimateAllTranslation(newXs, newYs, np.copy(img1), np.copy(img2))

	for k in range(len(bbox)):
		centers[k] = np.array([np.mean(bbox[k, :, 0]), np.mean(bbox[k, :, 1])]).astype(int)

	# Warp the image for the next iteration
	newXs, newYs, bbox, warped = applyGeometricTransformation(np.copy(x), np.copy(y), updatex, updatey, np.copy(orig_box), np.copy(img1), np.copy(img2), 0.4)

	for k in range(len(bbox)):
		xcen = int(np.mean(bbox[k, :, 0]))
		ycen = int(np.mean(bbox[k, :, 1]))
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

	frame = generate_output_frame(np.copy(img2), bbox, np.copy(trajectory_indexer))
	frame = Image.fromarray(frame)
	frame.save("medium_frame%d.jpg" % f)

	img1 = np.copy(img2)

cap.release()