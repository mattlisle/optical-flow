'''
  File name: objectTracking.py
  Author: 
  Date created: 11/16/2018

	(INPUT) rawVideo: The input video containing one or more objects
	(OUTPUT) trackedVideo: The generated output video showing all the tracked features (please do try to show
	the trajectories for all the features) on the object as well as the bounding boxes

'''

def objectTracking(rawVideo, output_filename="output.avi", draw_boxes=False):
	import cv2
	import numpy as np
	import matplotlib.pyplot as plt
	from helpers import rgb2gray
	from helpers import generate_output_frame
	from helpers import gen_video
	from getFeatures import getFeatures
	from calculateError import calculateError
	from estimateAllTranslation import estimateAllTranslation
	from applyGeometricTransformation import applyGeometricTransformation

	# Set parameters based on which video it is
	if rawVideo == "Easy.mp4":
		difficulty = "easy"
		print("Performing tracking on easy video")
		k_pad = 2
		params = [4, 1, 3, 1, 1.5]
	elif rawVideo == "Medium.mp4":
		difficulty = "medium"
		print("Performing tracking on medium video")
		k_pad = 2.5
		params = [3, 1, 3, 0, 0]
	else:
		print("Invalid path - valid videos are 'Easy.mp4' and 'Medium.mp4'")
		return None	

	imgs = np.array([])
	cap = cv2.VideoCapture(rawVideo)
	ret, img1 = cap.read()
	img1 = img1[...,::-1]
	h, w, d = img1.shape

	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	out = cv2.VideoWriter(output_filename, fourcc, 20.0, (w,h))

	# Load the bounding boxes that were drawn manually
	bbox = np.load("bbox_" + difficulty + ".npy")
	orig_box = np.copy(bbox)
	centers = np.zeros((len(bbox), 2))

	# For ploting the trajectory of the object
	trajectory_indexer = np.zeros((h, w), dtype=bool)

	# Get the features from inside the bounding box
	x, y = getFeatures(rgb2gray(img1), bbox)

	# Initialize these before the loop starts 
	newXs = np.copy(x)
	newYs = np.copy(y)

	# Record the initial frame
	f = 0
	frame = generate_output_frame(np.copy(img1), bbox, np.copy(trajectory_indexer), np.copy(newXs), np.copy(newYs))
	out.write(frame[..., ::-1])

	# Loop through the remainder of the frames
	while True:
		f += 1
		print("Processing frame: %d..." % f, end="\r", flush=True)

		if bbox.size:

			# Get new features every 8 frames and update the key frame
			if not f % 8:
				for i in range(len(bbox)):
					orig_box = np.copy(bbox)
				x, y = getFeatures(rgb2gray(img1), bbox)
				newXs = np.copy(x)
				newYs = np.copy(y)

			# Read the next frame
			ret, img2 = cap.read()
			if not ret:
				break

			# Switch to RGB
			img2 = img2[...,::-1]

			# Get the new feature locations in the next frame
			updatex, updatey, x, y = estimateAllTranslation(np.copy(newXs), np.copy(newYs), np.copy(x), np.copy(y), np.copy(img1), np.copy(img2), np.copy(bbox), params)

			# Find centers for trajectory plotting
			for k in range(len(bbox)):
				centers[k] = np.array([np.mean(bbox[k, :, 0]), np.mean(bbox[k, :, 1])]).astype(int)

			# Warp the image for the next iteration
			newXs, newYs, bbox = applyGeometricTransformation(np.copy(x), np.copy(y), updatex, updatey, np.copy(orig_box), np.copy(img1), np.copy(img2), k_pad)

			# Handle when we've gotten rid of a bounding box
			indexer = np.ones(len(bbox), dtype=bool)
			for k in range(len(bbox)):
				if not np.any(bbox[k]) or len(newXs[k]) < 2:
					indexer[k] = False

			# Nix everything associated with the removed bounding box
			bbox = bbox[indexer]
			orig_box = orig_box[indexer]
			newXs = newXs[indexer]
			newYs = newYs[indexer]
			x = x[indexer]
			y = y[indexer]
			centers = centers[indexer]

			# Plot the trajectory on the image
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

			# Generate the next frame
			frame = generate_output_frame(np.copy(img2), bbox, np.copy(trajectory_indexer), np.copy(newXs), np.copy(newYs))
			frame = frame[..., ::-1]
		
		# We have no bounding boxes, move on to generating the video
		else:
			ret, img2 = cap.read()
			if not ret:
				break
			frame = img2

		# Update img1 for the next loop
		img1 = np.copy(img2)
		out.write(frame)

	cap.release()
	out.release()

	return None