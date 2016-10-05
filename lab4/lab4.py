"""
CS4243 Lab 4
Name: Wu Yu Ting
Matric No.: A0118005W
"""
import sys, cv2
import numpy as np
import numpy.linalg as la

def get_gaussian_kernel(size, sigma=1.0):
	# returns a 2d gaussian kernel
	if size < 3:
		size = 3
	m = size/2
	x, y = np.mgrid[-m:m+1, -m:m+1]
	kernel = np.exp(-(x*x + y*y)/(2*sigma*sigma)) 
	kernel_sum = kernel.sum()

	if not sum == 0:
		kernel = kernel/kernel_sum 
	return kernel

def my_convolve(image, kernel):
	# Flip kernel
	kernel = np.fliplr(kernel)
	kernel = np.flipud(kernel)

	# Declare resulting image
	result = np.zeros(image.shape)

	# Find the strengths
	height, width = image.shape
	for i in xrange(1, height - 1):
		for j in xrange(1, width - 1):
			subset = image[i-1:i+2, j-1:j+2]
			result[i, j] = np.sum(kernel * subset)

	return result

def get_response_matrix(w_xx, w_xy, w_yy, step_size, k=0.06):
	# Declare response matrix
	height, width = w_xx.shape
	matrix = np.zeros([height/step_size, width/step_size])

	# Compute response values and store them in response matrix
	matrix_height, matrix_width = matrix.shape
	for i in xrange(matrix_height):
		for j in xrange(matrix_width):
			i_response = (i+1)*step_size-1
			j_response = (j+1)*step_size-1
			w = np.array([[w_xx[i_response, j_response], w_xy[i_response, j_response]], 
						 [w_xy[i_response, j_response], w_yy[i_response, j_response]]])
			det = la.det(w)
			trace = np.trace(w)
			response = det - k*trace*trace
			matrix[i, j] = response

	return matrix

def compute_corners(w_xx, w_xy, w_yy, step_size, threshold=0.9):
	response_matrix = get_response_matrix(w_xx, w_xy, w_yy, step_size)
	max_response = np.max(response_matrix)
	response_limit = max_response * (1-threshold)
	corners = []

	height, width = response_matrix.shape
	for i in xrange(height):
		for j in xrange(width):
			response = response_matrix[i, j]
			if response >= response_limit:
				corners.append(((i+1)*step_size-1, (j+1)*step_size-1))

	return corners

def plot_corners(image, corners):
	corner_image = np.array(image)
	for x, y in corners:
		cv2.rectangle(corner_image, (y-5, x-5), (y+5, x+5), (0,0,255), 1)
	return corner_image
	
SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
SOBEL_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
GAUSSIAN = get_gaussian_kernel(3)
FILE_PATH = sys.argv[1]

# Load image
image = cv2.imread(FILE_PATH, cv2.CV_LOAD_IMAGE_GRAYSCALE)

# Get strengths
gx = my_convolve(image, SOBEL_X)
gy = my_convolve(image, SOBEL_Y)

# Get product of derivatives
I_xx = gx * gx
I_xy = gx * gy
I_yy = gy * gy

# Get gaussian strengths on product of derivatives
W_xx = my_convolve(I_xx, GAUSSIAN)
W_xy = my_convolve(I_xy, GAUSSIAN)
W_yy = my_convolve(I_yy, GAUSSIAN)

# Get corners of size
corners_size1 = compute_corners(W_xx, W_xy, W_yy, 1)
corners_size10 = compute_corners(W_xx, W_xy, W_yy, 10)

# Plot corners on colour image
image = cv2.imread(FILE_PATH, cv2.CV_LOAD_IMAGE_COLOR)
image_size1 = plot_corners(image, corners_size1)
image_size10 = plot_corners(image, corners_size10)

# Save resulting images
file_name, file_extension = FILE_PATH.split('.')
cv2.imwrite(file_name + "_corner1." + file_extension, image_size1)
cv2.imwrite(file_name + "_corner10." + file_extension, image_size10)
