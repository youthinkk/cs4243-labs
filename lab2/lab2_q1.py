"""
CS4243 Lab 2 - Question 1
Name: Wu Yu Ting
Matric No.: A0118005W

Input command: python lab2_q1.py [image.jpg]
Output: image_hue.jpg image_saturation.jpg image_brightness.jpg
"""

import sys
import cv2
import numpy as np

MAX_INTENSITY = 255.0

# Conversion form RGB to HSV
def rgb_to_hsv(raw_red, raw_green, raw_blue):
	red, green, blue = raw_red / 255.0, raw_green / 255.0, raw_blue / 255.0
	cmax = max(red, green, blue)
	cmin = min(red, green, blue)
	cdiff = cmax - cmin

	#  compute hue
	if cdiff == 0:
		hue = 0
	elif cmax == red:
		hue = ((green - blue) / cdiff) % 6
	elif cmax == green:
		hue = ((blue - red) / cdiff) + 2
	elif cmax == blue:
		hue = ((red - green) / cdiff) + 4
	hue *= 60

	# compute saturation
	saturation = 0 if cmax == 0 else (cdiff / cmax)

	# compute brightness
	brightness = cmax

	return hue, saturation, brightness


FILE_PATH = sys.argv[1]

# Load image
image = cv2.imread(FILE_PATH)

# Get the image height and width
height, width, channel = image.shape

# Declare hue, saturation and brightness images
hue_image = np.zeros([height, width])
saturation_image = np.zeros([height, width])
brightness_image = np.zeros([height, width])

for index in np.ndindex(image.shape[:2]):
	# Get RGB of the pixel
	blue, green, red = image[index]

	# Compute hue, saturation and brightness
	hue, saturation, brightness = rgb_to_hsv(red, green, blue)

	# Normalise hue, saturation and brightness 
	hue_image[index] = int(round(hue * MAX_INTENSITY / 360, 0))
	saturation_image[index] = int(round(saturation * MAX_INTENSITY, 0))
	brightness_image[index] = int(round(brightness * MAX_INTENSITY, 0))

# Save resulting images
file_name, file_extension = FILE_PATH.split('.')
cv2.imwrite(file_name + "_hue." + file_extension, hue_image)
cv2.imwrite(file_name + "_saturation." + file_extension, saturation_image)
cv2.imwrite(file_name + "_brightness." + file_extension, brightness_image)