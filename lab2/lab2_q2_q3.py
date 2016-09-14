"""
CS4243 Lab 2 - Question 2 and 3
Name: Wu Yu Ting
Matric No.: A0118005W

Input command: python lab2_q2_q3.py [image_hue.jpg] [image_saturation.jpg] [image_brightness.jpg]
Output: image_hsv2rgb.jpg image_histeq.jpg
"""

import sys
import cv2
import numpy as np

INTENSITY_RANGE = 256
MAX_INTENSITY = 255.0

# Conversion from HSV to RGB
def hsv_to_rgb(hue, saturation, brightness):
	c = brightness * saturation
	x = c * (1 - abs((hue / 60.0) % 2 - 1))
	m = brightness - c

	section = int(hue / 60) % 6
	if section == 0: 
		red, green, blue = c, x, 0
	elif section == 1: 
		red, green, blue = x, c, 0
	elif section == 2: 
		red, green, blue = 0, c, x
	elif section == 3:
		red, green, blue = 0, x, c
	elif section == 4:
		red, green, blue = x, 0, c
	elif section == 5:
		red, green, blue = c, 0, x

	raw_red = int(round((red + m) * MAX_INTENSITY, 0))
	raw_green = int(round((green + m) * MAX_INTENSITY, 0))
	raw_blue = int(round((blue + m) * MAX_INTENSITY, 0))

	return [raw_blue, raw_green, raw_red]


# Histogram equalization
def hist_equalization(image):
	histogram = np.zeros(INTENSITY_RANGE)
	
	for index in np.ndindex(image.shape[:2]):
		pixel = image[index]
		histogram[pixel] += 1

	factor = MAX_INTENSITY / float(image.size)
	cum_histogram = [sum(histogram[:i+1]) for i in xrange(len(histogram))]
	norm_histogram = map(lambda x: int(round(x * factor, 0)), cum_histogram)

	new_image = np.array(map(lambda x: norm_histogram[x], image.flatten()))

	return new_image.reshape(image.shape)


HUE_FILE, SATURATION_FILE, BRIGHTNESS_FILE = sys.argv[1], sys.argv[2], sys.argv[3]

# Load hue, saturation and brightness images
hue_image = cv2.imread(HUE_FILE, cv2.CV_LOAD_IMAGE_GRAYSCALE)
saturation_image = cv2.imread(SATURATION_FILE, cv2.CV_LOAD_IMAGE_GRAYSCALE)
brightness_image = cv2.imread(BRIGHTNESS_FILE, cv2.CV_LOAD_IMAGE_GRAYSCALE)

# Histogram equalization on brightness image
equalized_brightness_image = hist_equalization(brightness_image)

# Get the image height and width
height, width = hue_image.shape

# Declare resulting RGB image before histogram equalization on brightness image
rgb_image = np.zeros([height, width, 3])

# Declare resulting RGB image after histogram equalization on brightness image
equalized_rgb_image = np.zeros([height, width, 3])

for index in np.ndindex(hue_image.shape[:2]):
	# Get the individual value of hue, saturation and brightness
	hue = hue_image[index] / MAX_INTENSITY * 360
	saturation = saturation_image[index] / MAX_INTENSITY
	brightness = brightness_image[index] / MAX_INTENSITY

	# Get equlized value of brightness
	equalized_brightness = equalized_brightness_image[index] / MAX_INTENSITY

	# Convert HSV to RGB
	rgb_image[index] = hsv_to_rgb(hue, saturation, brightness)
	equalized_rgb_image[index] = hsv_to_rgb(hue, saturation, equalized_brightness)

# Save resulting images
file_name, file_extension = HUE_FILE.split('_hue.')
cv2.imwrite(file_name + "_hsv2rgb." + file_extension, rgb_image)
cv2.imwrite(file_name + "_histeq." + file_extension, equalized_rgb_image)