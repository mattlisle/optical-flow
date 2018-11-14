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

frame1 = generate_output_frame(np.copy(img1), bbox)
frame1 = Image.fromarray(frame1)
frame1.save("easy_frame1.jpg")
# plt.imshow(frame1)
# plt.show()

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
# 	plt.scatter(x[i], y[i][:], color="blue")
# plt.show()

warped = np.copy(img1)
newXs = np.copy(x)
newYs = np.copy(y)
iterations = 3
for k in range(iterations):
	# Get the new feature locations in the next frame
	updatex, updatey = estimateAllTranslation(newXs, newYs, warped, img2)

	# Warp the image for the next iteration
	newXs, newYs, bbox, warped = applyGeometricTransformation(newXs, newYs, updatex, updatey, bbox, warped)

frame2 = generate_output_frame(np.copy(img2), bbox)
frame2 = Image.fromarray(frame2)
frame2.save("easy_frame2.jpg")

# For debugging: Show the bounding box and the features inside
plt.imshow(img1)
for box in bbox:
	for i in range(3):
		plt.plot(box[i: i+2, 0], box[i: i+2, 1], color="red")
	plt.plot([box[0, 0], box[3, 0]], [box[0, 1], box[3, 1]], color="red")
for i in range(len(x)):
	plt.scatter(x[i], y[i], color="blue")
	plt.scatter(newXs[i][:], newYs[i][:], color="green")
plt.show()
