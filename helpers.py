'''
  File name: helpers.py
  Author: Nikhil, Shiv, Matt
  Date created: 11/4/2018
'''

def rgb2gray(rgb):
    import numpy as np
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


def anms(cimg, max_pts, offsetx, offsety):
  import numpy as np
  from time import time
  from scipy import signal

  # Initialize array of minimum radii
  minimum_r = np.ones(cimg.shape)
  
  # Will be needing these later
  h, w = cimg.shape

  # ---------- Part 1: Find points that are local maxima --------- #
  # Define 4 kernels that will allow us to compare to 4-nearest pixel neighbors
  # Switched from 0.9 to 1 and got better performance
  left  = np.array([ 0, 1, -1]).reshape(1, 3)
  right = np.array([-1, 1,  0]).reshape(1, 3)
  up    = np.array([ 0, 1, -1]).reshape(3, 1)
  down  = np.array([-1, 1,  0]).reshape(3, 1)

  # Generate comparison array, one array along 0th dim for each neighbor
  comps = np.zeros((4, h, w))
  comps[0:, ...] = signal.convolve2d(cimg,  left, mode="same")
  comps[1:, ...] = signal.convolve2d(cimg, right, mode="same")
  comps[2:, ...] = signal.convolve2d(cimg,    up, mode="same")
  comps[3:, ...] = signal.convolve2d(cimg,  down, mode="same")

  # Use comps to create 2d array of local maxima
  max_locs = np.all(comps > 0, axis=0)

  # ---------- Part 2: Loop through all points and find radii ---------- #
  # Initialize x and y with locations where points clear 4 nearest neighbors
  y, x = np.where(max_locs)
  values = cimg[max_locs]

  # Debugging print statement
  total = cimg.shape[0] * cimg.shape[1]
  maxes = len(values)
  print("%d maxima from %d points" % (maxes, total))

  # Sort these values in decreasing order
  sorter = np.argsort(-values)
  x = x[sorter]
  y = y[sorter]
  values = values[sorter]

  # Initialize array of radii for each interest point, already know value for first pt
  radii = np.zeros(len(values))
  radii[0] = np.nan_to_num(np.Inf)

  # Compute the Euclidean distance of every interest point to every other interest point
  distances = np.zeros(len(values))
  for i in range(1, len(values)):
    distances = np.sqrt(np.square(x[:i] - x[i]) + np.square(y[:i] - y[i]))
    radii[i] = np.amin(distances)

  # ---------- Part 3: Construct outputs based on max_pts ---------- #
  sorter = np.argsort(-radii)
  x = x[sorter]
  y = y[sorter]
  radii = radii[sorter]

  # If we've asked for more than we've got, let the user know
  if max_pts > len(x):
    print("Actual number of points: " + str(len(x)))

  # Otherwise cut out the fat and index the max radius
  else:
    x = x[:max_pts]
    y = y[:max_pts]

  return x + offsetx, y + offsety
