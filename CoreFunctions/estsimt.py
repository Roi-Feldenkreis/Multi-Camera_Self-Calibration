import numpy as np
from scipy.linalg import norm
from scipy.linalg import svd


def estsimt(X1, X2):
    """
    Estimate similarity transformation

    Parameters:
    X1 (ndarray): 3xN matrix with corresponding 3D points
    X2 (ndarray): 3xN matrix with corresponding 3D points

    Returns:
    s (float): Scalar scale
    R (ndarray): 3x3 rotation matrix
    T (ndarray): 3x1 translation vector
    """
    N = X1.shape[1]

    if N != X2.shape[1]:
        raise ValueError('estsimt: both sets must contain the same number of points')

    X1cent = np.mean(X1, axis=1, keepdims=True)
    X2cent = np.mean(X2, axis=1, keepdims=True)

    # Normalize coordinate systems for both set of points
    x1 = X1 - X1cent
    x2 = X2 - X2cent

    # Mutual distances
    d1 = x1[:, 1:] - x1[:, :-1]
    d2 = x2[:, 1:] - x2[:, :-1]
    ds1 = norm(d1, axis=0)
    ds2 = norm(d2, axis=0)

    scales = ds2 / ds1

    # The scales should be the same for all points, use median to reduce noise effect
    s = np.median(scales)

    # Undo scale
    x1s = s * x1

    # Finding rotation
    H = np.zeros((3, 3))
    for i in range(N):
        H += np.outer(x1s[:, i], x2[:, i])

    U, S, Vt = svd(H)
    V = Vt.T
    R = V @ U.T

    T = X2cent - s * R @ X1cent

    return s, R, T
