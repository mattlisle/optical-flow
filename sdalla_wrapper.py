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
h, w, d = img1.shape

# For now, manually draw the bounding box and forget about cv2.boundingRect()
box1 = np.array([287, 187, 397, 187, 397, 264, 287, 264]).reshape(4, 2)
box2 = np.array([223, 123, 277, 123, 277, 168, 223, 168]).reshape(4, 2)
bbox = np.array([box1, box2])
# bbox = np.array([box1])
orig_box = np.copy(bbox)
centers = np.zeros((len(bbox), 2))
trajectory_indexer = np.zeros((h, w), dtype=bool)
'''
f = 0
frame = generate_output_frame(np.copy(img1), bbox, np.copy(trajectory_indexer))
frame = Image.fromarray(frame)
frame.save("easy_frame%d.jpg" % f)
'''

# Get the features from inside the bounding box, but use opencv
botx = min(box1[:,0])
topx = max(box1[:,0])
boty = min(box1[:,1])
topy = max(box1[:,1])
gray = cv2.cvtColor(img1[botx:topx,boty:topy], cv2.COLOR_BGR2GRAY)
pts = cv2.goodFeaturesToTrack(gray, 25,0.1,5)

print(pts.shape)

x = pts[:,:,0]
y = pts[:,:,1]
print(x)
print(y)
# For debugging: Show the bounding box and the features inside
fig, (left, right) = plt.subplots(1, 2)
left.imshow(img1)
for box in bbox:
	for i in range(3):
		left.plot(box[i: i+2, 0], box[i: i+2, 1], color="red")
	left.plot([box[0, 0], box[3, 0]], [box[0, 1], box[3, 1]], color="red")
for i in range(len(x)):
	left.scatter(x[i]+botx, y[i]+boty, color="blue")
plt.show()
'''
a = 0
while f < 99:
	f += 1
	a += 1
	if not f % 10:
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
'''