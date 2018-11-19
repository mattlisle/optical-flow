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
import argparse

refPt = []
def draw_box(event, x, y, flags, param):
  # grab references to the global variables
  global refPt, cropping
  
  # if the left mouse button was clicked, record the starting
  # (x, y) coordinates and indicate that cropping is being
  # performed
  if event == cv2.EVENT_LBUTTONDOWN:
    refPt.append((x,y))
  
  # check to see if the left mouse button was released
  elif event == cv2.EVENT_LBUTTONUP:
    # record the ending (x, y) coordinates and indicate that
    # the cropping operation is finished
    refPt.append((x, y))
  
  # draw a rectangle around the region of interest
  # cv2.rectangle(display_img, refPt[0], refPt[1], (0, 255, 0), 2)
  # cv2.imshow("Start Window", display_img)



def main(video_file, output_filename): 
	imgs = np.array([])
	cap = cv2.VideoCapture(video_file)
	ret, img1 = cap.read()
	img1 = img1[...,::-1]
	h, w, d = img1.shape

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

	bboxes = []
	for i in range(int(len(refPt)/2)):
		bbox = []
		start = refPt[2*i]
		end = refPt[(2*i) + 1]
		# Top Left and bottom right
		box_corners = np.array([refPt[2*i], refPt[(2*i)+1]])
		start_row, start_col, width, height = cv2.boundingRect(box_corners)

		bbox = np.array([[start_row, start_col],
							[start_row, start_col + width],
							[start_row+height, start_col + width],
							[start_row+height, start_col]])

		bboxes.append(bbox)
				




	# For now, manually draw the bounding box and forget about cv2.boundingRect()
	# box1 = np.array([456, 182, 456, 279, 523, 279, 523, 182]).reshape(4, 2)
	# bbox = np.array([box1])

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
	# frame.save("easy_frame%d.jpg" % f)

	# Store the processed frames so we can turn it into a video later
	all_frames = []
	all_frames.append(frame)

	a = 0
	while ret:
		f += 1
		a += 1
		if not f % 8:
			print("Frame: ", f)
			a = 1
			for i in range(len(bbox)):
				# xmin = np.sort(bbox[i, :, 0])[0]
				# xmax = np.sort(bbox[i, :, 0])[3]
				# ymin = np.sort(bbox[i, :, 1])[0]
				# ymax = np.sort(bbox[i, :, 1])[3]
				# bbox[i, ...] = np.array([xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]).reshape(4,2)
				orig_box = np.copy(bbox)
			x, y = getFeatures(rgb2gray(img1), bbox)
			newXs = np.copy(x)
			newYs = np.copy(y)

		thresh = .1 + .02 * a

		ret, img2 = cap.read()
		if not ret:
			break
		img2 = img2[...,::-1]

		iterations = 1

		# Get the new feature locations in the next frame
		updatex, updatey, x, y = estimateAllTranslation(newXs, newYs, np.copy(x), np.copy(y), np.copy(img1), np.copy(img2), np.copy(bbox))

		for k in range(len(bbox)):
			centers[k] = np.array([np.mean(bbox[k, :, 0]), np.mean(bbox[k, :, 1])]).astype(int)

		# Warp the image for the next iteration
		newXs, newYs, bbox, warped = applyGeometricTransformation(np.copy(x), np.copy(y), updatex, updatey, np.copy(orig_box), np.copy(img1), np.copy(img2), thresh)

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

		frame = generate_output_frame(np.copy(img2), bbox, np.copy(trajectory_indexer), np.copy(newXs), np.copy(newYs))
		frame = Image.fromarray(frame)
		# frame.save("medium_frame%d.jpg" % f)

		img1 = np.copy(img2)
		all_frames.append(frame)

	cap.release()

	np_frames = np.array([np.array(f) for f in all_frames])
	gen_video(np.array(np_frames), "{0}.avi".format(output_filename))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("-V", "--video", required=True, help="name of the MP4 file to analyze")
    parser.add_argument("-O", "--outputVideo", required=True, help="name of the output video without extension (AVI file)")
    
    args = vars(parser.parse_args())

    video_file = args['video']
    output_filename = args['outputVideo']
    main(video_file, output_filename)
