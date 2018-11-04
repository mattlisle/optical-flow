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

cap = cv2.VideoCapture("Easy.mp4")
ret, img = cap.read()
cap.release()

img = img[...,::-1]

# For now, manually draw the bounding box and forget about cv2.boundingRect()
# In the format of [xmin, xmax, ymin, ymax]
bbox = np.array([287, 397, 187, 264])

# For debugging: Show the bounding box we've chosen
# plt.imshow(img)
# plt.plot([bbox[0], bbox[0], bbox[1], bbox[1], bbox[0]], [bbox[2], bbox[3], bbox[3], bbox[2], bbox[2]], color="red")
# plt.show()

# Get the features from inside the bounding box
colorimg = np.copy(img)
img = rgb2gray(img)
x, y = getFeatures(img, bbox)

# For debugging: Show the bounding box and the features inside
plt.imshow(colorimg)
plt.plot([bbox[0], bbox[0], bbox[1], bbox[1], bbox[0]], [bbox[2], bbox[3], bbox[3], bbox[2], bbox[2]], color="red")
plt.scatter(x, y, color="blue")
plt.show()
