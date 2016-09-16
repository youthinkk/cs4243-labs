"""
CS4243 Lab 3
Name: Wu Yu Ting
Matric No.: A0118005W
"""

import sys, cv2, math
import numpy as np

SOBEL = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
PREWIT = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
MAX_INTENSITY = 255

def calc_strength(kernel_x, kernel_y, target):
	strength_x = np.sum(kernel_x * target)
	strength_y = np.sum(kernel_y * target)
	return math.sqrt(strength_x**2 + strength_y**2)

def my_convolve(image, ff):
	# Flip kernel
	ff = np.fliplr(ff)
	ff = np.flipud(ff)

	# Get kernel for x and y directions
	kernel_x = np.transpose(ff)
	kernel_y = ff

	# Declare resulting image
	height, width = image.shape
	result = np.zeros([height, width])

	# Find the edges
	for i in xrange(1, height - 1):
		for j in xrange(1, width - 1):
			subset = image[i-1:i+2, j-1:j+2]
			strength = calc_strength(kernel_x, kernel_y, subset)
			result[i, j] = strength

	# Normalise image
	min_strength = min(result.flatten())
	max_strength = max(result.flatten())
	result = np.round((result - min_strength) / (max_strength - min_strength) * MAX_INTENSITY).astype(int)

	return result

def thinning(image):
	result = np.zeros(image.shape)
	height, width = image.shape

	for i in xrange(1, height - 1):
		for j in xrange(1, width - 1):
			pixel = image[i, j]
			max_x = max(pixel, image[i, j-1], image[i, j+1])
			max_y = max(pixel, image[i-1, j], image[i+1, j])
			result[i, j] = pixel if (pixel == max_x or pixel == max_y) else 0
	return result

FILE_PATH = sys.argv[1]

# Load image
image = cv2.imread(FILE_PATH, cv2.CV_LOAD_IMAGE_GRAYSCALE)

# Get edges
sobel_result = my_convolve(image, SOBEL)
prewit_result = my_convolve(image, PREWIT)
thinning_result = thinning(sobel_result)

# Save resulting images
file_name, file_extension = FILE_PATH.split('.')
cv2.imwrite(file_name + "_sobel." + file_extension, sobel_result)
cv2.imwrite(file_name + "_prewit." + file_extension, prewit_result)
cv2.imwrite(file_name + "_thinning." + file_extension, thinning_result)