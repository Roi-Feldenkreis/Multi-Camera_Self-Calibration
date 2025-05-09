import numpy as np


def fsampson(F, u):
    """
    Calculate first order geometrical error (Sampson Distance)

    Parameters:
    -----------
    F : numpy.ndarray (3x3)
        Fundamental matrix
    u : numpy.ndarray (6xN)
        Point pairs in homogeneous coordinates
        First 3 rows are points in first image
        Last 3 rows are points in second image

    Returns:
    --------
    errs : numpy.ndarray (1xN)
        Sampson distance for each point pair

    Notes:
    ------
    The Sampson distance is an approximation to the geometric error
    that's more efficient to compute than the true geometric error.
    """

    N = u.shape[1]

    u1 = u[0:3, :]  # Points in first image
    u2 = u[3:6, :]  # Points in second image

    errs = np.zeros(N)

    for i in range(N):
        Fu1 = F @ u1[:, i]  # F * u1
        Fu2 = F.T @ u2[:, i]  # F' * u2

        # Calculate the Sampson distance
        # (u2.T * F * u1)^2 / (sum of squares of first two components of Fu1 and Fu2)
        numerator = (u2[:, i].T @ F @ u1[:, i]) ** 2
        denominator = np.sum(np.concatenate(([Fu1[0:2]], [Fu2[0:2]])) ** 2)

        errs[i] = numerator / denominator

    return errs