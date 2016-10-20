"""
CS4243 Lab 6
Name: Wu Yu Ting
Matric No.: A0118005W
"""
import math
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

# projection parameters
U_0 = 0
V_0 = 0
BETA_U = 1
BETA_V = 1
K_U = 1
K_V = 1
FOCAL_LENGTH = 1

# constant
FRAME_SIZE = 4
PLOT_ROW_SIZE = 2
PLOT_COLUMN_SIZE = 2


def init_points():
    points = np.zeros([11, 3])
    points[0, :] = [-1, -1, -1]
    points[1, :] = [1, -1, -1]
    points[2, :] = [1, 1, -1]
    points[3, :] = [-1, 1, -1]
    points[4, :] = [-1, -1, 1]
    points[5, :] = [1, -1, 1]
    points[6, :] = [1, 1, 1]
    points[7, :] = [-1, 1, 1]
    points[8, :] = [-0.5, -0.5, -1]
    points[9, :] = [0.5, -0.5, -1]
    points[10, :] = [0, 0.5, -1]
    return points


def conjugate(q):
    s = q[0:1]
    v = [-1 * value for value in q[1:]]
    return s + v


# quaternion multiplication
def quatmult(q1, q2):
    s1, v1 = q1[0], np.array(q1[1:])
    s2, v2 = q2[0], np.array(q2[1:])
    s = s1 * s2 - np.dot(v1, v2)
    v = np.cross(v1, v2) + s1 * v2 + s2 * v1

    out = [0] * 4
    out[0], out[1:] = s, v
    return out


# performs rotation
# p: point quaternion, q: rotation quaternion
def compute_rotation_quat(p, q):
    return quatmult(quatmult(q, p), conjugate(q))


# computes a 3x3 rotation matrix parameterized
# by the elements of a given input quaternion
def quat2rot(q):
    w, x, y, z = q[0], q[1], q[2], q[3]
    rotation_matrix = [
        [1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - w * z), 2 * (w * y + w * z)],
        [2 * (x * y + w * z), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x ** 2 + y ** 2)]]
    return np.matrix(rotation_matrix)


def orthographic_projection(scene, translation, i, j):
    diff = scene - translation
    u = np.dot(diff, i) * BETA_U + U_0
    v = np.dot(diff, j) * BETA_V + V_0
    return [u, v]


def perspective_projection(scene, translation, i, j, k):
    diff = scene - translation
    u = FOCAL_LENGTH * np.dot(diff, i) / np.dot(diff, k) * BETA_U + U_0
    v = FOCAL_LENGTH * np.dot(diff, j) / np.dot(diff, k) * BETA_V + V_0
    return [u, v]


def plot_figure(projected_points, title):
    figure = plt.figure()
    plt.subplots_adjust(wspace=0.3, hspace=0.4)
    for i in xrange(FRAME_SIZE):
        frame = i + 1
        points = projected_points[i]
        plt.subplot(PLOT_ROW_SIZE, PLOT_COLUMN_SIZE, frame)
        plt.title("Frame " + str(frame), fontsize=12)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.axis("equal")
        plt.margins(0.1)

        for point in points:
            plt.plot(point[0], point[1], "go")

    file_name = "_".join(title.lower().split())
    plt.suptitle(title, fontsize=16)
    figure.savefig(file_name + ".png")


def compute_homography(src, dst):
    matrix = []
    for i in xrange(src.shape[0]):
        x, y, z = src[i]/src[i, 2]
        u, v = dst[i]
        matrix.append([x, y, z, 0, 0, 0, -u*x, -u*y, -u*z])
        matrix.append([0, 0, 0, x, y, z, -v*x, -v*y, -v*z])

    U, s, V = la.svd(matrix)
    H = V[-1].reshape(3, 3)
    H = H / H[2, 2]

    return H

############
# Part 1.2 #
############
angle = -30
initial_position = [0, 0, 0, -5]
camera_positions = [initial_position]
rotation_quat = [math.cos(math.radians(angle / 2)), 0, math.sin(math.radians(angle / 2)), 0]

# compute camera positions for subsequent 3 frames
for i in xrange(FRAME_SIZE-1):
    position = np.array(compute_rotation_quat(camera_positions[i], rotation_quat))
    camera_positions.append(position)

print "rotation_quat:\n", np.array(rotation_quat)
print "camera_positions:\n", np.array(camera_positions)
print

############
# Part 1.3 #
############
angle = 30
initial_orientation = np.identity(3)
camera_orientations = [initial_orientation]
rotation_matrix = quat2rot([math.cos(math.radians(angle / 2)), 0, math.sin(math.radians(angle / 2)), 0])

# compute camera orientations for subsequent 3 frames
for i in xrange(FRAME_SIZE-1):
    orientation = np.array(np.dot(rotation_matrix, camera_orientations[i]))
    camera_orientations.append(orientation)

print "rotation_matrix:\n", np.array(rotation_matrix)
print "camera_orientations:\n", np.array(camera_orientations)
print

##########
# Part 2 #
##########
orthographic_points = []
perspective_points = []
points = init_points()

# compute orthographic and perspective projected points
for i in xrange(FRAME_SIZE):
    translation = camera_positions[i][1:]
    orientation_i = camera_orientations[i][0]
    orientation_j = camera_orientations[i][1]
    orientation_k = camera_orientations[i][2]

    orthographic_points.append([orthographic_projection(p, translation, orientation_i, orientation_j)
                                for p in points])
    perspective_points.append([perspective_projection(p, translation, orientation_i, orientation_j, orientation_k)
                               for p in points])

plot_figure(orthographic_points, "Orthographic Projection")
plot_figure(perspective_points, "Perspective Projection")

print "orthographic projection:\n", np.array(orthographic_points)
print "perspective projection:\n", np.array(perspective_points)
print

##########
# Part 3 #
##########
src = np.array([points[0], points[1], points[2], points[3], points[8]])
dst = np.array([perspective_points[2][0], perspective_points[2][1], perspective_points[2][2],
                perspective_points[2][3], perspective_points[2][8]])

homography = compute_homography(src, dst)
print "homography:\n", homography
print

for i in xrange(src.shape[0]):
    computed_dst = np.dot(homography, np.transpose(src[i]))
    print "computed_dst: ", computed_dst / computed_dst[-1]
    print "truth_dst: ", dst[i]
    print
