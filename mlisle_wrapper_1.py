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
cap = cv2.VideoCapture("Easy.mp4")
ret, img1 = cap.read()
img1 = img1[...,::-1]

# For now, manually draw the bounding box and forget about cv2.boundingRect()
box1 = np.array([287, 187, 397, 187, 397, 264, 287, 264]).reshape(4, 2)
box2 = np.array([223, 123, 277, 123, 277, 168, 223, 168]).reshape(4, 2)
# bbox = np.array([box1, box2])
bbox = np.array([box1])
orig_box = np.copy(bbox)

f = 0
frame = generate_output_frame(np.copy(img1), bbox)
frame = Image.fromarray(frame)
frame.save("easy_frame%d.jpg" % f)


# Get the features from inside the bounding box
x, y = getFeatures(rgb2gray(img1), bbox)

newXs = np.copy(x)
newYs = np.copy(y)

# For debugging: Show the bounding box and the features inside
# fig, (left, right) = plt.subplots(1, 2)
# left.imshow(img1)
# for box in bbox:
# 	for i in range(3):
# 		left.plot(box[i: i+2, 0], box[i: i+2, 1], color="red")
# 	left.plot([box[0, 0], box[3, 0]], [box[0, 1], box[3, 1]], color="red")
# for i in range(len(x)):
# 	left.scatter(x[i], y[i], color="blue")
a = 0
while f < 159:
	f += 1
	a += 1
	if not f % 10:
		a = 1
		for i in range(len(bbox)):
			xmin = np.amin(bbox[i, :, 0])
			xmax = np.amax(bbox[i, :, 0])
			ymin = np.amin(bbox[i, :, 1])
			ymax = np.amax(bbox[i, :, 1])
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

	# Warp the image for the next iteration
	newXs, newYs, bbox, warped = applyGeometricTransformation(np.copy(x), np.copy(y), updatex, updatey, np.copy(orig_box), np.copy(img1), np.copy(img2), 0.4)

	frame = generate_output_frame(np.copy(img2), bbox)
	frame = Image.fromarray(frame)
	frame.save("easy_frame%d.jpg" % f)

	img1 = np.copy(img2)

cap.release()

# right.imshow(img2)

# xshift = np.mean((newXs - x)[0])
# yshift = np.mean((newYs - y)[0])

# newXs, newYs, bbox, warped = applyGeometricTransformation(x, y, newXs, newYs, np.copy(orig_box), np.copy(img1), np.copy(img2), .4)

# for box in orig_box:
# 	for i in range(3):
# 		right.plot(box[i: i+2, 0], box[i: i+2, 1], color="red")
# 	right.plot([box[0, 0], box[3, 0]], [box[0, 1], box[3, 1]], color="red")
# for box in bbox:
# 	for i in range(3):
# 		right.plot(box[i: i+2, 0], box[i: i+2, 1], color="orange")
# 	right.plot([box[0, 0], box[3, 0]], [box[0, 1], box[3, 1]], color="orange")
# for i in range(len(x)):
# 	right.scatter(x[i], y[i], color="blue")
# 	right.scatter(newXs[i][:], newYs[i][:], color="green")
# plt.show()
