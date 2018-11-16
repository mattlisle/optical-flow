'''
  File name: objectTracking.py
  Author: 
  Date created: 11/16/2018

	(INPUT) rawVideo: The input video containing one or more objects
	(OUTPUT) trackedVideo: The generated output video showing all the tracked features (please do try to show
	the trajectories for all the features) on the object as well as the bounding boxes

'''

def objectTracking(rawVideo, difficulty="easy"):
	import cv2
	import numpy as np
	import matplotlib.pyplot as plt
	from PIL import Image
	from helpers import rgb2gray
	from helpers import generate_output_frame
	from getFeatures import getFeatures
	from estimateAllTranslation import estimateAllTranslation
	from applyGeometricTransformation import applyGeometricTransformation

	if difficulty == "easy":
		# For now, manually draw the bounding box and forget about cv2.boundingRect()
		box1 = np.array([287, 187, 397, 187, 397, 264, 287, 264]).reshape(4, 2)
		box2 = np.array([223, 123, 277, 123, 277, 168, 223, 168]).reshape(4, 2)
		bbox = np.array([box1, box2])
		# bbox = np.array([box1])
		orig_box = np.copy(bbox)
	else:
		print("We're not there yet")
		return None


	imgs = np.array([])
	cap = cv2.VideoCapture(rawVideo)
	ret, img1 = cap.read()
	img1 = img1[...,::-1]

	f = 0
	frame = generate_output_frame(np.copy(img1), bbox)
	frame = Image.fromarray(frame)
	frame.save("easy_frame%d.jpg" % f)


	# Get the features from inside the bounding box
	x, y = getFeatures(rgb2gray(img1), bbox)

	newXs = np.copy(x)
	newYs = np.copy(y)
	
	a = 0
	while f < 159:
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

		# Warp the image for the next iteration
		newXs, newYs, bbox, warped = applyGeometricTransformation(np.copy(x), np.copy(y), updatex, updatey, np.copy(orig_box), np.copy(img1), np.copy(img2), 0.4)

		frame = generate_output_frame(np.copy(img2), bbox)
		frame = Image.fromarray(frame)
		frame.save("easy_frame%d.jpg" % f)

		img1 = np.copy(img2)

	cap.release()