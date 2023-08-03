"""
Author: Pit Henrich
Date: 2023-07

Compute the best-fit ellipsoid of a set of 3D points.
"""

import numpy as np

def fit_ellipsoid(points):
    """
    Parameters
    ----------
    input_array : numpy.ndarray
        (n, 3) array, where 'n' is the number of points, and
        the second dimension consists of the 'x', 'y', and 'z' coordinates of each point.
        You will need at least n=10 points to fit an ellipsoid.

    Returns
    -------
    centre : numpy.ndarray
        (3,) array containing the centre of the ellipsoid
    eigenvectors : numpy.ndarray
        (3, 3) array, each column contains the corresponding eigenvector
    radii : numpy.ndarray
        (3,) array containing the radii of the ellipsoid
    """
    # Construct design matrix
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    #             A     B     C     D      E      F      G    H    I    J
    D = np.array([x**2, y**2, z**2, 2*x*y, 2*x*z, 2*y*z, 2*x, 2*y, 2*z, np.ones_like(x)]).T

    # Find solution in column space of the design matrix (algebraic solution for least squares problem)
    # (singular vector corresponding to smallest singular value)
    _, _, V = np.linalg.svd(D, full_matrices=False)

    # No need to normalize now, do a homogeneous normalization later
    # Ax²  + By²  + Cz²  + 2Dxy  + 2Exz  + 2Fyz  + 2Gx  + 2Hy  + 2Iz  + J = 0
    # <=>
    # A'x² + B'y² + C'z² + 2D'xy + 2E'xz + 2F'yz + 2G'x + 2H'y + 2I'z + 1 = 0
    v = V[-1, :]

    # Matrix representation of algebraic form:
    #   x^T Q x = Ax² + By² + Cz² + 2Dxy + 2Exz + 2Fyz + 2Gx + 2Hy + 2Iz + J = 0
    #   =>
    #   Q   = [A, D, E, G]
    #         [D, B, F, H]
    #         [E, F, C, I]
    #         [G, H, I, J]
    Q = np.array([[v[0], v[3], v[4], v[6]],
                  [v[3], v[1], v[5], v[7]],
                  [v[4], v[5], v[2], v[8]],
                  [v[6], v[7], v[8], v[9]]])

    # Centre of the ellipsoid (find linear terms)
    # Solve:
    #   Q[:3,:3]c + Q[:3, 3] = 0
    #   <=>
    #   Q[:3,:3]c = -Q[:3, 3]
    #   <=>
    #   Ax + Dy + Ez = -G
    #   Dx + By + Fz = -H
    #   Ex + Fy + Cz = -I
    centre = np.linalg.solve(-Q[:3, :3], Q[:3, 3])

    # Form the corresponding translation matrix
    T = np.eye(4)
    T[:3, 3] = centre

    # Translate to the center
    R = T.T @ Q @ T

    # Homogeneous normalization
    R = R / R[3, 3]

    # Get eigenvectors/values of centred ellipsoid to describe deformation
    eigenvalues, eigenvectors = np.linalg.eig(R[:3, :3])

    # Radii = 1 / ||eigenvector * eigenvalue||_2
    radii = np.sqrt(1. / np.abs(eigenvalues))

    return centre, eigenvectors, radii
