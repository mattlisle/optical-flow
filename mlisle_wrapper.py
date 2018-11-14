'''
  File name: mlisle_wrapper.py
  Author: Matt
  Date created: 11/4/2018
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt
from helpers import rgb2gray
from getFeatures import getFeatures
from estimateAllTranslation import estimateAllTranslation

cap = cv2.VideoCapture("Easy.mp4")
ret, img1 = cap.read()
ret, img2 = cap.read()
cap.release()

img1 = img1[...,::-1]
img2 = img2[...,::-1]

# For now, manually draw the bounding box and forget about cv2.boundingRect()
box1 = np.array([287, 187, 397, 187, 397, 264, 287, 264]).reshape(4, 2)
box2 = np.array([223, 123, 277, 123, 277, 168, 223, 168]).reshape(4, 2)
bbox = np.array([box1, box2])

# For debugging: Show the bounding box we've chosen
# plt.imshow(img1)
# for box in bbox:
# 	for i in range(3):
# 		plt.plot(box[i: i+2, 0], box[i: i+2, 1], color="red")
# 	plt.plot([box[0, 0], box[3, 0]], [box[0, 1], box[3, 1]], color="red")
# plt.show()

# Get the features from inside the bounding box
x, y = getFeatures(rgb2gray(img1), bbox)

# For debugging: Show the bounding box and the features inside
# plt.imshow(img1)
# for box in bbox:
# 	for i in range(3):
# 		plt.plot(box[i: i+2, 0], box[i: i+2, 1], color="red")
# 	plt.plot([box[0, 0], box[3, 0]], [box[0, 1], box[3, 1]], color="red")
# for i in range(x.shape[1]):
# 	plt.scatter(x[:, i], y[:, i], color="blue")
# plt.show()

# Get the new feature locations in the next frame
newXs, newYs = estimateAllTranslation(x, y, img1, img2)

# For debugging: Show the bounding box and the features inside
plt.imshow(img1)
for box in bbox:
	for i in range(3):
		plt.plot(box[i: i+2, 0], box[i: i+2, 1], color="red")
	plt.plot([box[0, 0], box[3, 0]], [box[0, 1], box[3, 1]], color="red")
for i in range(x.shape[1]):
	plt.scatter(x[:, i], y[:, i], color="blue")
	plt.scatter(newXs[:, i], newYs[:, i], color="green")
plt.show()
