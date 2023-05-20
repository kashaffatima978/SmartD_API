# import the necessary packages
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2

def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def preprocess_image(path):
  # load the image, convert it to grayscale, and blur it slightly
  image = cv2.imread(path)
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# gray = cv2.GaussianBlur(gray, (7, 7), 0)
# cv2_imshow(gray)
# perform edge detection, then perform a dilation + erosion to
# close gaps in between object edges
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  blur = cv2.GaussianBlur(gray, (9, 9), 0)

  edged = cv2.Canny(blur, 50, 100)
  edged = cv2.dilate(edged, None, iterations=1)
  edged = cv2.erode(edged, None, iterations=1)
# find contours in the edge map
  cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
  cnts = imutils.grab_contours(cnts)

  (cnts, _) = contours.sort_contours(cnts)
  cntsArea = []
  for c in cnts:
      cntsArea.append(cv2.contourArea(c))
  return cnts , cntsArea

def order_points(pts):
	# sort the points based on their x-coordinates
	xSorted = pts[np.argsort(pts[:, 0]), :]
	# grab the left-most and right-most points from the sorted
	# x-roodinate points
	leftMost = xSorted[:2, :]
	rightMost = xSorted[2:, :]
	# now, sort the left-most coordinates according to their
	# y-coordinates so we can grab the top-left and bottom-left
	# points, respectively
	leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
	(tl, bl) = leftMost
	# now that we have the top-left coordinate, use it as an
	# anchor to calculate the Euclidean distance between the
	# top-left and right-most points; by the Pythagorean
	# theorem, the point with the largest distance will be
	# our bottom-right point
	D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
	(br, tr) = rightMost[np.argsort(D)[::-1], :]
	# return the coordinates in top-left, top-right,
	# bottom-right, and bottom-left order
	return np.array([tl, tr, br, bl], dtype="float32")

def estimate(cnts,cntsArea, path):
  pixelsPerMetric = None
  image = cv2.imread(path)
  largeObj = np.argmax(cntsArea)
  c = cnts[largeObj]
# compute the rotated bounding box of the contour
  orig = image.copy()
  box = cv2.minAreaRect(c)
  box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
  box = np.array(box, dtype="int")
  print(box)
# order the points in the contour such that they appear
# in top-left, top-right, bottom-right, and bottom-left
# order, then draw the outline of the rotated bounding
# box
  box = order_points(box)
  cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
# loop over the original points and draw them
  for (x, y) in box:
    cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)
  # unpack the ordered bounding box, then compute the midpoint
	# between the top-left and top-right coordinates, followed by
	# the midpoint between bottom-left and bottom-right coordinates
  (tl, tr, br, bl) = box
  (tltrX, tltrY) = midpoint(tl, tr)
  (blbrX, blbrY) = midpoint(bl, br)
	# compute the midpoint between the top-left and top-right points,
	# followed by the midpoint between the top-righ and bottom-right
  (tlblX, tlblY) = midpoint(tl, bl)
  (trbrX, trbrY) = midpoint(tr, br)
	# draw the midpoints on the image
  cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
  cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
  cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
  cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
	# draw lines between the midpoints
  cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
	(255, 0, 255), 2)
  cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
	(255, 0, 255), 2)
 	# compute the Euclidean distance between the midpoints
  dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
  dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
  print('DA', dA, "DB", dB)
	# if the pixels per metric has not been initialized, then
	# compute it as the ratio of pixels to supplied metric
	# (in this case, inches)
  if pixelsPerMetric is None:
    pixelsPerMetric = dB / 0.955
  	# compute the size of the object
  dimA = dA / pixelsPerMetric
  dimB = dB / pixelsPerMetric

  print(dimA)
  print(dimB)
	# draw the object sizes on the image
  cv2.putText(orig, "{:.1f}in".format(dimA),
	(int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
	0.65, (255, 255, 255), 2)
  cv2.putText(orig, "{:.1f}in".format(dimB),
		(int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
		0.65, (255, 255, 255), 2)
	# show the output image
  # cv2.imshow(orig)
  cv2.waitKey(0)
  return dimA, dimB
