"""
Author: Pit Henrich
Date: 2023-07

Compute the best-fit ellipsoid of a set of 3D points.
"""

import numpy as np
from ellipsoid_fit import fit_ellipsoid
import random


def sample_sphere(r, c, num_samples):
    # Generate random spherical coordinates
    phi = np.random.uniform(0, np.pi*2, num_samples)  # azimuthal angle
    costheta = np.random.uniform(-1, 1, num_samples)  # cos(theta), where theta is the polar angle

    theta = np.arccos(costheta)  # convert back to theta

    # Convert spherical coordinates to Cartesian coordinates
    x = np.abs(r * np.sin(theta) * np.cos(phi)) + c[0]
    y = np.abs(r * np.sin(theta) * np.sin(phi)) + c[1]
    z = np.abs(r * np.cos(theta)) + c[2]


    # Stretch the sphere into an ellipsoid
    x *= random.uniform(0.5, 2.0)
    y *= random.uniform(0.5, 2.0)
    z *= random.uniform(0.5, 2.0)

    # Rotate the ellipsoid
    theta = np.deg2rad(random.uniform(0, 360))
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))
    x, y, z = np.dot(np.array([x, y, z]).T, R).T

    # Rotate along the y-axis
    theta = np.deg2rad(random.uniform(0, 360))
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, 0, s), (0, 1, 0), (-s, 0, c)))
    x, y, z = np.dot(np.array([x, y, z]).T, R).T

    # Rotate along the z-axis
    theta = np.deg2rad(random.uniform(0, 360))
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))
    x, y, z = np.dot(np.array([x, y, z]).T, R).T

    return np.vstack([x, y, z]).T

# Set parameters for sphere
r = random.uniform(0.5, 100)
c = np.array([random.uniform(-100, 100), random.uniform(-100, 100), random.uniform(-100, 100)])
num_samples = 10

points = sample_sphere(r, c, num_samples)

c, directions, radii = fit_ellipsoid(points)

print('Center:\n', c)
print('Directions:\n', directions)
print('Radii:\n', radii)


# Plot the results
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot points
ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='g', marker='.', s=100)

# Plot center of ellipsoid
ax.scatter(c[0], c[1], c[2], color='r', marker='o', s=100)

# Sample 100 points on the surface of a sphere
u = np.linspace(0, 2 * np.pi, 30)
v = np.linspace(0, np.pi, 30)
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones(np.size(u)), np.cos(v))


# Construct transformation matrix from directions and radii
T = np.eye(4)
T[:3, :3] = directions * radii
T[3, 3] = 1.0
T[:3, 3] = c

# Transform sphere points
x, y, z, _ = np.dot(T, np.array([x.flatten(), y.flatten(), z.flatten(), np.ones_like(x.flatten())]))

# Plot sphere points
ax.scatter(x, y, z, color='b', marker='.', s=10)

# Ensure aspect ratio is 1 so sphere appears as a sphere
ax.set_aspect('equal')

plt.show()
