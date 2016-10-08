"""
CS4243 Lab 5
Name: Wu Yu Ting
Matric No.: A0118005W
"""

import cv2, sys
import cv2.cv as cv
import numpy as np

FILE_PATH = sys.argv[1]

# Load video capture
capture = cv2.VideoCapture(FILE_PATH)

# Get frame parameters
frame_width = int(capture.get(cv.CV_CAP_PROP_FRAME_WIDTH))
frame_height = int(capture.get(cv.CV_CAP_PROP_FRAME_HEIGHT))
frame_rate = int(capture.get(cv.CV_CAP_PROP_FPS))
frame_count = int(capture.get(cv.CV_CAP_PROP_FRAME_COUNT))

# Print frame parameters
print "Frame width: ", frame_width
print "Frame height: ", frame_height
print "Frame rate: ", frame_rate
print "Frame count: ", frame_count

# Get background
_, frame = capture.read()
average_image = np.float32(frame)
for i in xrange(1, frame_count):
	alpha = 1. / (i+1)
	_, frame = capture.read()
	cv2.accumulateWeighted(frame, average_image, alpha)

	print "fr = ", i, "alpha = ", alpha

# Normalise the background image
norm_image = np.uint8(np.round(average_image))

# Save resulting images
file_name, file_extension = FILE_PATH.split('.')
cv2.imwrite(file_name + "_background.jpg", norm_image)
