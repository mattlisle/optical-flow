'''
  File name: helpers.py
  Author: Nikhil, Shiv, Matt
  Date created: 11/4/2018
'''

def rgb2gray(rgb):
    import numpy as np
    if len(rgb.shape) == 2:
      return rgb
    else:
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
  pad = 5
  max_locs = np.all(comps > 0, axis=0)[pad:-pad, pad:-pad] # Not letting in points 10 pixels or closer to boundary
  max_locs = np.pad(max_locs, ((pad, pad), (pad, pad)), mode="constant")

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
  # x = x[sorter]
  # y = y[sorter]
  # values = values[sorter]

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


def inlier_cost_func(H, x, y):
  import numpy as np

  H = np.stack([H[:3], H[3:6], np.array([0, 0, 1])])
  # H = H.reshape(3, 3)
  estimates = np.matmul(H, x)
  residuals = y - estimates / estimates[2]

  h, num_inliers = x.shape

  return residuals.reshape(h * num_inliers)

def inlier_cost_func_translation(t, x, y):
  import numpy as np

  H = np.identity(3)
  H[0, 2] = t[0]
  H[1, 2] = t[1]
  estimates = np.matmul(H, x)
  residuals = y - estimates

  h, num_inliers = x.shape

  return residuals.reshape(h * num_inliers)

def warp_image(img, H, xmin, xmax, ymin, ymax):
  import numpy as np
  from scipy.ndimage import map_coordinates
  import matplotlib.pyplot as plt

  x, y = np.meshgrid(np.arange(xmin, xmax), np.arange(ymin, ymax))
  h, w = x.shape

  # Assignment suggested geometric_transform for this part, but I dunno how to use it, so I'll just brute-force vectorize
  pts = np.stack([x.reshape(h*w), y.reshape(h*w), np.ones(h*w)])

  Hinv = np.linalg.inv(H)
  Hinv = Hinv / Hinv[-1, -1]

  transformed = np.zeros((3, h * w))
  transformed = np.matmul(Hinv, pts)
  transformed = transformed / transformed[2]

  warped = map_coordinates(img, [transformed[1], transformed[0]]).reshape(h, w).astype(float)

  return warped

def generate_output_frame(img, bbox, indexer):
  import numpy as np

  h, w, d = img.shape

  for i in range(bbox.shape[0]):
    xmin = np.amin(bbox[i, :, 0])
    xmax = np.amax(bbox[i, :, 0])
    ymin = np.amin(bbox[i, :, 1])
    ymax = np.amax(bbox[i, :, 1])

    # indexer = np.zeros((h, w), dtype=bool)
    indexer[ymin: ymin + 2, xmin: xmax + 1] = True
    indexer[ymax - 1: ymax + 1, xmin: xmax + 1] = True
    indexer[ymin: ymax + 1, xmin: xmin + 2] = True
    indexer[ymin: ymax + 1, xmax - 1: xmax + 1] = True

    img[..., 0][indexer] = 255
    img[..., 1][indexer] = 0
    img[..., 2][indexer] = 0

  return img


def interp2(v, xq, yq):
  import numpy as np
  import pdb


  if len(xq.shape) == 2 or len(yq.shape) == 2:
    dim_input = 2
    q_h = xq.shape[0]
    q_w = xq.shape[1]
    xq = xq.flatten()
    yq = yq.flatten()

  h = v.shape[0]
  w = v.shape[1]
  if xq.shape != yq.shape:
    raise 'query coordinates Xq Yq should have same shape'


  x_floor = np.floor(xq).astype(np.int32)
  y_floor = np.floor(yq).astype(np.int32)
  x_ceil = np.ceil(xq).astype(np.int32)
  y_ceil = np.ceil(yq).astype(np.int32)

  x_floor[x_floor<0] = 0
  y_floor[y_floor<0] = 0
  x_ceil[x_ceil<0] = 0
  y_ceil[y_ceil<0] = 0

  x_floor[x_floor>=w-1] = w-1
  y_floor[y_floor>=h-1] = h-1
  x_ceil[x_ceil>=w-1] = w-1
  y_ceil[y_ceil>=h-1] = h-1

  v1 = v[y_floor, x_floor]
  v2 = v[y_floor, x_ceil]
  v3 = v[y_ceil, x_floor]
  v4 = v[y_ceil, x_ceil]

  lh = yq - y_floor
  lw = xq - x_floor
  hh = 1 - lh
  hw = 1 - lw

  w1 = hh * hw
  w2 = hh * lw
  w3 = lh * hw
  w4 = lh * lw

  interp_val = v1 * w1 + w2 * v2 + w3 * v3 + w4 * v4

  if dim_input == 2:
    return interp_val.reshape(q_h,q_w)
  return interp_val

def est_affine(x1, x2):
  import numpy as np

  A = np.zeros((6, 6))

  A[:3] = np.pad(x1.T, ((0, 0), (0, 3)), mode="constant")
  A[3:] = np.pad(x1.T, ((0, 0), (3, 0)), mode="constant")

  b = np.concatenate((x2[0], x2[1])).reshape(6, 1)

  try:
    Ainv = np.linalg.inv(A)
    T = np.concatenate((np.matmul(Ainv, b).reshape(2, 3), np.array([[0, 0, 1]])))
  except np.linalg.LinAlgError:
    T = np.identity(3)

  return T