'''
  File name: objectTracking.py
  Author: 
  Date created: 11/16/2018

	(INPUT) rawVideo: The input video containing one or more objects
	(OUTPUT) trackedVideo: The generated output video showing all the tracked features (please do try to show
	the trajectories for all the features) on the object as well as the bounding boxes

'''
import argparse
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
import math

# Global variable used for drawing the bounding boxes on the image
refPt = []

def draw_box(event, x, y, flags, param):
  # grab references to the global variables
  global refPt
  
  # if the left mouse button was clicked, record the starting (x, y) coordinates
  if event == cv2.EVENT_LBUTTONDOWN:
    refPt.append((x,y))
  
  # check to see if the left mouse button was released
  elif event == cv2.EVENT_LBUTTONUP:
    # record the ending (x, y) coordinates 
    refPt.append((x, y))
  

def objectTracking(rawVideo, output_filename, draw_boxes=False):
	imgs = np.array([])
	cap = cv2.VideoCapture(rawVideo)
	ret, img1 = cap.read()
	img1 = img1[...,::-1]
	h, w, d = img1.shape

	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	out = cv2.VideoWriter("{0}.avi".format(output_filename), fourcc, 20.0, (w,h))

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

	if draw_boxes:
		display_img = img1.copy()
		display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
		cv2.namedWindow("Start Frame")
		cv2.setMouseCallback("Start Frame", draw_box)

		# Loop until the user is done drawing boxes
		while True:
				cv2.imshow("Start Frame", display_img)
				key = cv2.waitKey(0)

				if key == ord('q'):
					break

		# Destroy the drawing window
		cv2.destroyAllWindows()

		# Show the result
		for i in range(int(len(refPt)/2)):
				cv2.rectangle(display_img, refPt[2*i], refPt[(2*i)+1], (0,255,0), 2)

		cv2.imshow("Result", display_img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

		bbox = []
		for i in range(int(len(refPt)/2)):

				# Top Left and bottom right
				box_corners = np.array([refPt[2*i], refPt[(2*i)+1]])
				start_x, start_y, width, height = cv2.boundingRect(box_corners)

				# Create the four coordinates for the box and reshape		
				box = np.array([[start_x, start_y],
										[start_x+width, start_y],
										[start_x+width, start_y + height],
										[start_x, start_y + height]])

				bbox.append(box)
			

		# Turn it into a numpy array
		bbox = np.array(bbox)
	else:
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
	while f < 100:
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

			# Build Pyramid 1
			L = np.copy(img1)
			pyramid1 = [L]
			for i in range(3):
				h1, w1, c1 = pyramid1[i].shape
				next_h, next_w = math.floor((h1 + 1)/2), math.floor((w1+1)/2)

				# Pad image 1
				p1 = np.zeros((h1+2, w1+2, 3))
				p1[1:h1+1, 1:w1+1, :] = pyramid1[i]
				p1[1:h1+1,0,:] = pyramid1[i][0:h1,0,:]
				p1[1:h1+1,w1+1,:] = pyramid1[i][0:h1,w1-1,:]
				p1[0,1:w1+1,:] = pyramid1[i][0,0:w1,:]
				p1[h1+1,1:w1+1,:] = pyramid1[i][h1-1,0:w1,:]
				p1[0,0,:] = (p1[0,1,:]+ p1[1,0,:]+ p1[1,1,:])/3
				p1[0,w1+1,:] = (p1[0,w1,:]+ p1[1,w1+1,:]+ p1[1,w1,:])/3
				p1[h1+1,w1+1,:] = (p1[h1,w1,:]+ p1[h1,w1+1,:]+ p1[h1+1,w1,:])/3
				p1[h1+1,0,:] = (p1[h1,0,:]+ p1[h1,1,:]+ p1[h1+1,1,:])/3


				hh, ww = np.meshgrid(np.arange(next_h), np.arange(next_w))

				next_level = np.zeros((next_h, next_w, 3))

				next_level = (.25 * p1[2*hh, 2*ww,:]) + (1/8) * (p1[2*hh-1,2*ww,:] + p1[2*hh+1,2*ww,:] + p1[2*hh,2*ww-1,:] + p1[2*hh,2*ww+1,:]) + (1/16) * (p1[2*hh-1,2*ww-1,:] + p1[2*hh+1,2*ww+1,:] + p1[2*hh-1,2*ww+1,:] + p1[2*hh+1,2*ww+1,:])

				pyramid1.append(next_level)

			for i in range(1,len(pyramid1),2):
				pyramid1[i] = np.transpose(pyramid1[i], axes=(1,0,2))

			for i in range(len(pyramid1)):
				cv2.imwrite('Temp{0}.png'.format(i), pyramid1[i])


			# Build Pyramid 2
			L = np.copy(img2)
			pyramid2 = [L]
			for i in range(3):
				h2, w2, c2 = pyramid2[i].shape
				next_h, next_w = math.floor((h2 + 1)/2), math.floor((w2+1)/2)

				# Pad image 1
				p2 = np.zeros((h2+2, w2+2, 3))
				p2[1:h2+1, 1:w2+1, :] = pyramid2[i]
				p2[1:h2+1,0,:] = pyramid2[i][0:h2,0,:]
				p2[1:h2+1,w2+1,:] = pyramid2[i][0:h2,w2-1,:]
				p2[0,1:w2+1,:] = pyramid2[i][0,0:w2,:]
				p2[h2+1,1:w2+1,:] = pyramid2[i][h2-1,0:w2,:]
				p2[0,0,:] = (p2[0,1,:]+ p2[1,0,:]+ p2[1,1,:])/3
				p2[0,w2+1,:] = (p2[0,w2,:]+ p2[1,w2+1,:]+ p2[1,w2,:])/3
				p2[h2+1,w2+1,:] = (p2[h2,w2,:]+ p2[h2,w2+1,:]+ p2[h2+1,w2,:])/3
				p2[h2+1,0,:] = (p2[h2,0,:]+ p2[h2,1,:]+ p2[h2+1,1,:])/3


				hh, ww = np.meshgrid(np.arange(next_h), np.arange(next_w))

				next_level = np.zeros((next_h, next_w, 3))

				next_level = (.25 * p2[2*hh, 2*ww,:]) + (1/8) * (p2[2*hh-1,2*ww,:] + p2[2*hh+1,2*ww,:] + p2[2*hh,2*ww-1,:] + p2[2*hh,2*ww+1,:]) + (1/16) * (p2[2*hh-1,2*ww-1,:] + p2[2*hh+1,2*ww+1,:] + p2[2*hh-1,2*ww+1,:] + p2[2*hh+1,2*ww+1,:])

				pyramid2.append(next_level)

			for i in range(1,len(pyramid2),2):
				pyramid2[i] = np.transpose(pyramid2[i], axes=(1,0,2))

			for i in range(len(pyramid1)):
				cv2.imwrite('temp{0}.png'.format(i), pyramid2[i])

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


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("-V", "--video", required=True, help="name of the MP4 file to analyze")
    parser.add_argument("-O", "--outputVideo", required=True, help="name of the output video without extension (AVI file)")
    parser.add_argument("-b", "--boundingBox", required=False, action='store_true', help="include if you want to draw the bounding boxes manually instead of using prebuilt ones")
    
    args = vars(parser.parse_args())

    video_file = args['video']
    output_filename = args['outputVideo']
    draw_bounding_box = args['boundingBox']
    objectTracking(video_file, output_filename, draw_bounding_box)
