import numpy as np


def pointnormalization(u):
    """
    Isotropic point normalization for improved numerical stability

    Parameters:
    -----------
    u : numpy.ndarray (3xN)
        Input data points in homogeneous coordinates

    Returns:
    --------
    u2 : numpy.ndarray (3xN)
        Normalized data points in homogeneous coordinates
    T : numpy.ndarray (3x3)
        Transformation matrix that performs the normalization

    Notes:
    ------
    This normalization improves the numerical stability of linear methods
    for estimating geometric transformations. It:
        1. Translates points so their centroid is at the origin
        2. Scales points so their average distance from origin is sqrt(2)
    """

    n = u.shape[1]

    # Calculate mean of x and y coordinates
    xmean = np.mean(u[0, :])
    ymean = np.mean(u[1, :])

    # Create a copy of the input data
    u2 = u.copy()

    # Center the points (translate centroid to origin)
    u2[0:2, :] = u[0:2, :] - np.tile(np.array([[xmean], [ymean]]), (1, n))

    # Calculate scale factor so average distance is sqrt(2)
    scale = np.sqrt(2) / np.mean(np.sqrt(np.sum(u2[0:2, :] ** 2, axis=0)))

    # Apply scaling
    u2[0:2, :] = scale * u2[0:2, :]

    # Create the transformation matrix
    T = np.diag([scale, scale, 1.0])
    T[0, 2] = -scale * xmean
    T[1, 2] = -scale * ymean

    return u2, T