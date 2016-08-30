"""
CS4243 Lab 1
Name: Wu Yu Ting
Matric No.: A0118005W
"""

import numpy as np
import numpy.linalg as la

# Read the data
file = open("data.txt")
data = np.genfromtxt(file, delimiter=",")
file.close()

# Format M
M_flatten = np.apply_along_axis(lambda arr: np.append(np.append(np.append(arr, 1), np.zeros(6)), np.append(arr, 1)), 1, data[:, 2:]).flatten()
M = np.matrix(M_flatten.reshape(-1, 6))

# Format b
b_flatten = data[:, :2].flatten()
print b_flatten
b = np.matrix(b_flatten.reshape(-1, 1))

# Compute least-squared error solution
a, e, r, s = la.lstsq(M, b)

# Print outputs
print "data = \n", data
print
print "M = \n", M
print
print "b = \n", b
print
print "Least-squared solution = \n", a
print
print "Sum-squared error = ", la.norm(M * a - b) ** 2
print
print "Residue = ", e
print