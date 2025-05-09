import numpy as np


def isptnorm(x):
    """
    Isotropic Point Normalization

    Normalizes point coordinates to improve numerical stability of geometric algorithms.
    This normalization translates points to have zero mean and scales them to have
    a specific average distance from the origin.

    Parameters:
    -----------
    x : numpy.ndarray (N x dim)
        Input coordinates, where N is the number of points and dim is the dimensionality
        (typically 2 for image points or 3 for 3D points)

    Returns:
    --------
    xnorm : numpy.ndarray (N x dim)
        Normalized coordinates
    T : numpy.ndarray ((dim+1) x (dim+1))
        Transformation matrix that performs the normalization

    Notes:
    ------
    The transformation includes:
    1. Translation of the centroid to the origin
    2. Scaling so that the average distance from origin is sqrt(dimension)

    This normalization is important for numerical stability in algorithms like DLT
    (Direct Linear Transform) for estimating homographies or fundamental matrices.
    """

    # Get data dimensions
    N, dim = x.shape

    # Make homogeneous coordinates (add a column of ones)
    x_homog = np.hstack((x, np.ones((N, 1))))

    # Compute sum of squared differences from mean for each dimension
    ssd = np.zeros((N, dim))
    for i in range(dim):
        ssd[:, i] = (x[:, i] - np.mean(x[:, i])) ** 2

    # Compute the scaling factor
    # The goal is to make the average distance from origin = sqrt(dim)
    scale = (np.sqrt(dim) * N) / np.sum(np.sqrt(np.sum(ssd.T)))

    # Create the transformation matrix
    T = np.zeros((dim + 1, dim + 1))

    # Set scaling factors on diagonal
    for i in range(dim):
        T[i, i] = scale
        # Set translation components (last column)
        T[i, dim] = -scale * np.mean(x[:, i])

    # Set bottom-right element to 1 (homogeneous part)
    T[dim, dim] = 1

    # Apply transformation
    xnorm = T @ x_homog.T
    xnorm = xnorm.T

    # Return only the non-homogeneous part
    return xnorm[:, 0:dim], T