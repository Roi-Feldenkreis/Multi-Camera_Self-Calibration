import numpy as np
from RANSAC_Algorithm.point_normalization import pointnormalization


def u2fdlt(u, do_norm=True):
    """
    Linear estimation of the Fundamental matrix from point correspondences using DLT.
    Fully vectorized, uses thin SVD for speed.
    """

    u = u.T
    no_points = u.shape[0]

    if no_points < 8:
        raise ValueError('Too few correspondences. At least 8 points are required.')

    if u.shape[1] == 4:
        ones = np.ones((no_points, 1))
        u = np.hstack((u[:, 0:2], ones, u[:, 2:4], ones))

    u1 = u[:, 0:3]
    u2 = u[:, 3:6]

    if do_norm:
        u1_t, T1 = pointnormalization(u1.T)
        u1 = u1_t.T
        u2_t, T2 = pointnormalization(u2.T)
        u2 = u2_t.T

    # Vectorized design matrix (replaces triple for-loop)
    A = np.einsum('ij,ik->ijk', u2, u1).reshape(no_points, 9)

    # Thin SVD is faster than full SVD (only compute needed singular vectors)
    _, _, Vh = np.linalg.svd(A, full_matrices=False)
    F = Vh[-1, :].reshape(3, 3)   # No .T: row-major reshape already gives correct F

    if do_norm:
        F = np.linalg.inv(T2) @ F @ T1

    return F
