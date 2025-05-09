import numpy as np
from point_normalization import pointnormalization


def u2fdlt(u, do_norm=True):
    """
    Linear estimation of the Fundamental matrix from point correspondences using DLT

    Parameters:
    -----------
    u : numpy.ndarray (6xN or 4xN)
        Point correspondences. If 6xN, points are already in homogeneous coordinates.
        If 4xN, points are in Cartesian coordinates and will be converted to homogeneous.
    do_norm : bool, optional (default=True)
        Whether to apply isotropic normalization to improve numerical stability

    Returns:
    --------
    F : numpy.ndarray (3x3)
        Fundamental matrix

    Notes:
    ------
    This implements the Direct Linear Transform (DLT) algorithm for estimating
    the fundamental matrix from point correspondences. At least 8 point pairs
    are required.
    """

    # Transpose for easier handling (make points as rows)
    u = u.T

    no_points = u.shape[0]

    if no_points < 8:
        raise ValueError('Too few correspondences. At least 8 points are required.')

    # Check if input is in Cartesian coordinates and convert to homogeneous if needed
    if u.shape[1] == 4:
        # Convert to homogeneous coordinates by adding ones
        ones = np.ones((no_points, 1))
        u = np.hstack((u[:, 0:2], ones, u[:, 2:4], ones))

    # Split the points into the two views
    u1 = u[:, 0:3]  # Points in first image
    u2 = u[:, 3:6]  # Points in second image

    # Apply isotropic normalization if requested
    if do_norm:
        u1_t, T1 = pointnormalization(u1.T)
        u1 = u1_t.T

        u2_t, T2 = pointnormalization(u2.T)
        u2 = u2_t.T

    # Create the design matrix A
    A = np.zeros((no_points, 9))

    for i in range(no_points):
        for j in range(3):
            for k in range(3):
                A[i, j * 3 + k] = u2[i, j] * u1[i, k]

    # Perform SVD to find the solution
    U, S, Vh = np.linalg.svd(A)

    # The solution is the last column of V (transpose of Vh)
    f = Vh.T[:, -1]

    # Reshape the solution to a 3x3 matrix
    F = f.reshape(3, 3).T

    # Denormalize if normalization was applied
    if do_norm:
        F = np.linalg.inv(T2) @ F @ T1

    return F