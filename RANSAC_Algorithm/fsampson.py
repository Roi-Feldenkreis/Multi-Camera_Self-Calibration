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
    errs : numpy.ndarray (N,)
        Sampson distance for each point pair

    Notes:
    ------
    Fully vectorized - no Python for loops over points.
    """

    u1 = u[0:3, :]  # 3xN - Points in first image
    u2 = u[3:6, :]  # 3xN - Points in second image

    Fu1  = F   @ u1   # 3xN : F * u1
    FTu1 = F.T @ u1   # 3xN : F' * u1  (NOTE: uses u1, matching MATLAB)

    # Numerator: (u2' * F * u1)^2  ->  shape (N,)
    numerator = np.sum(u2 * Fu1, axis=0) ** 2

    # Denominator: sum of squares of first 2 rows of Fu1 and FTu1  ->  shape (N,)
    denominator = (np.sum(Fu1[:2, :]  ** 2, axis=0) +
                   np.sum(FTu1[:2, :] ** 2, axis=0))

    errs = numerator / denominator

    return errs
